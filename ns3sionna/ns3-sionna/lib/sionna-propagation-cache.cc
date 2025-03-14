/*
 * Copyright (c) 2024 Yannik Pilz
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Yannik Pilz <y.pilz@campus.tu-berlin.de>
 */

#include "sionna-propagation-cache.h"

#include "message.pb.h"
#include "sionna-mobility-model.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SionnaPropagationCache");

NS_OBJECT_ENSURE_REGISTERED(SionnaPropagationCache);

TypeId
SionnaPropagationCache::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SionnaPropagationCache")
            .SetParent<Object>()
            .SetGroupName("Propagation")
            .AddConstructor<SionnaPropagationCache>();
    return tid;
}

SionnaPropagationCache::SionnaPropagationCache()
    : m_sionnaHelper(nullptr), m_caching(true), m_cache_hits(0), m_cache_miss(0), m_optimize(true)
{
    m_friisLossModel = CreateObject<FriisPropagationLossModel>();
    m_constSpeedDelayModel = CreateObject<ConstantSpeedPropagationDelayModel>();
}

SionnaPropagationCache::~SionnaPropagationCache()
{
    m_cache.clear();
}

Time
SionnaPropagationCache::GetPropagationDelay(Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
    // Check if distance is too far so that a simpler model can be used
    if (m_optimize)
    {
        const double MAX_TXPOWER_DBM = 20.0; // AZU: todo: hardcoded
        double resultdBm = m_friisLossModel->CalcRxPower(MAX_TXPOWER_DBM, a, b);
        if (resultdBm + m_optimize_margin < m_sionnaHelper->GetNoiseFloor())
        {
            Time const_delay = m_constSpeedDelayModel->GetDelay(a, b);
            NS_LOG_INFO("Skipped raytracing for prop delay due to large distance; const delay used: " << const_delay);
            return const_delay;
        }
        // signal is too strong and delay need to be computed with ray tracing
    }

    return GetPropagationData(a, b).m_delay;
}

double
SionnaPropagationCache::GetPropagationLoss(Ptr<MobilityModel> a, Ptr<MobilityModel> b, double txPowerDbm) const
{
    // Check if distance is too far so that a simpler model can be used
    if (m_optimize)
    {
        double resultdBm = m_friisLossModel->CalcRxPower(txPowerDbm, a, b);
        if (resultdBm + m_optimize_margin < m_sionnaHelper->GetNoiseFloor())
        {
            double friis_loss = (-1) * (resultdBm - txPowerDbm);
            NS_LOG_INFO("Skipped raytracing for prop loss due to large distance; friis loss used: " << friis_loss);
            return friis_loss;
        }
        // signal is too strong and need to be computed with ray tracing
    }
    return GetPropagationData(a, b).m_loss;
}

void
SionnaPropagationCache::SetSionnaHelper(SionnaHelper &sionnaHelper)
{
    m_sionnaHelper = &sionnaHelper;
}

void
SionnaPropagationCache::SetCaching(bool caching)
{
    m_caching = caching;
}

void
SionnaPropagationCache::SetOptimize(bool optimize)
{
    m_optimize = optimize;
}

double
SionnaPropagationCache::GetStats()
{
    double ratio = m_cache_hits / (m_cache_hits + m_cache_miss);
    return ratio;
}

SionnaPropagationCache::CacheEntry
SionnaPropagationCache::GetPropagationData(Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
    NS_ASSERT_MSG(m_sionnaHelper, "SionnaPropagationCache must have reference to SionnaHelper.");
    Ptr<SionnaMobilityModel> sionna_a = DynamicCast<SionnaMobilityModel> (a);
    Ptr<SionnaMobilityModel> sionna_b = DynamicCast<SionnaMobilityModel> (b);
    NS_ASSERT_MSG(sionna_a && sionna_b, "Not using SionnaMobilityModel.");

    Time current_time = Simulator::Now();

    Ptr<Node> node_a = a->GetObject<Node>();
    Ptr<Node> node_b = b->GetObject<Node>();
    NS_ASSERT_MSG(node_a && node_b, "Nodes not found.");

    NS_LOG_INFO("GetPropagationData:: " << node_a->GetId() << " to " << node_b->GetId());

    if (m_caching)
    {
        //NS_LOG_INFO("Check cache");
        // Look in the cache and check if delay and loss have already been calculated for the two nodes
        CacheKey key = CacheKey(node_a->GetId(), node_b->GetId());
        auto it = m_cache.find(key);

        if (it != m_cache.end())
        {
            //NS_LOG_INFO(" Cache entry found but need to check if expired ");
            // remove outdated entries
            std::vector<CacheEntry>::iterator vec_it = it->second.begin();
            while (vec_it != it->second.end()) {
                if (vec_it->m_end_time < current_time) {
                    vec_it = it->second.erase(vec_it);
                } else {
                    ++vec_it;
                }
            }

            // iterate over all stored entries for that link
            for (CacheEntry c_entry : it->second) {
                //NS_LOG_INFO(" " << c_entry.m_start_time << " " << c_entry.m_end_time << " " << current_time);
                // If delay and loss exist in the cache, check if the entry is not outdated
                if (c_entry.m_end_time >= current_time && c_entry.m_start_time <= current_time)
                {
                    NS_LOG_INFO("Cache HIT CSI:: " << node_a->GetId() << " to " << node_b->GetId());
                    m_cache_hits += 1;
                    // Return cache entry as the value is still fresh
                    return c_entry;
                }
            }
        }
    }

    NS_LOG_INFO("Cache MISS CSI:: " << node_a->GetId() << " to " << node_b->GetId());
    m_cache_miss += 1;

    // Prepare the request message
    ns3sionna::Wrapper wrapper;

    // Fill the information message
    ns3sionna::ChannelStateRequest* propagation_request = wrapper.mutable_channel_state_request();
    propagation_request->set_tx_node(node_a->GetId());
    propagation_request->set_rx_node(node_b->GetId());
    propagation_request->set_time(current_time.GetNanoSeconds());
    
    // Serialize the request message
    std::string serialized_message;
    wrapper.SerializeToString(&serialized_message);

    std::cout << "SEND channel_state_request ----> " << std::endl;
    sleep(3);
    // Send the request message
    zmq::message_t zmq_message(serialized_message.data(), serialized_message.size());
    m_sionnaHelper->m_zmq_socket.send(zmq_message, zmq::send_flags::none);

    // Receive the reply message
    zmq::message_t zmq_reply;
    zmq::recv_result_t result = m_sionnaHelper->m_zmq_socket.recv(zmq_reply, zmq::recv_flags::none);
    
    std::cout << "<----------------- REPLY channel_state_request" << std::endl;

    NS_ASSERT_MSG(result, "Failed to receive reply after propagation request message.");

    // Check if the reply message is a propagation response
    ns3sionna::Wrapper reply_wrapper;
    reply_wrapper.ParseFromArray(zmq_reply.data(), zmq_reply.size());
    //NS_LOG_INFO("ZMQ::CSI_RESP sz=" << zmq_reply.size() << " Bytes");

    NS_ASSERT_MSG(reply_wrapper.has_channel_state_response(), "Reply after channel state request is not a channel state response.");
    std::cout << "Reply after channel state request is not a channel state response." << std::endl;

    // Extract the delay, loss and time to live value
    const ns3sionna::ChannelStateResponse& csi_response = reply_wrapper.channel_state_response();

    NS_LOG_INFO("ZMQ::CSI_RESP #samples: " << csi_response.csi_size());
    
    // result contains also future CSI; fill-up the cache
    for (int csi_i=0; csi_i < csi_response.csi_size(); csi_i++) {
        Time start_time = NanoSeconds(csi_response.csi(csi_i).start_time());
        Time end_time = NanoSeconds(csi_response.csi(csi_i).end_time());

        NS_LOG_INFO("CSI TS: " << start_time << " - " << end_time);

        google::protobuf::uint32 txId = csi_response.csi(csi_i).tx_node().id();

        for (int rx_i=0; rx_i < csi_response.csi(csi_i).rx_nodes_size(); rx_i++) {
            Time delay = NanoSeconds(csi_response.csi(csi_i).rx_nodes(rx_i).delay());
            double wb_loss = csi_response.csi(csi_i).rx_nodes(rx_i).wb_loss();
            //Time ttl = NanoSeconds(csi_response.csi(csi_i).rx_nodes(rx_i).ttl());

            google::protobuf::uint32 rxId = csi_response.csi(csi_i).rx_nodes(rx_i).id();

            NS_LOG_INFO("    -> sionna Response (delay: " << delay << ", loss: " << wb_loss << ")"
              << " (TxId: " << txId << " [" << csi_response.csi(csi_i).tx_node().position().x()
              << "," << csi_response.csi(csi_i).tx_node().position().y() << "," << csi_response.csi(csi_i).tx_node().position().z() << "] -> "
              << rxId << " [" << csi_response.csi(csi_i).rx_nodes(rx_i).position().x() << "," << csi_response.csi(csi_i).rx_nodes(rx_i).position().y()
              << "," << csi_response.csi(csi_i).rx_nodes(rx_i).position().z() << "])");

            // CSI
            std::stringstream ss;
            ss << "[";
            for (int i=0; i < csi_response.csi(csi_i).rx_nodes(rx_i).csi_imag().size(); i++)
            {
                double imag = csi_response.csi(csi_i).rx_nodes(rx_i).csi_imag(i);
                double real = csi_response.csi(csi_i).rx_nodes(rx_i).csi_real(i);
                ss << imag << "*1i + " << real;

                if (i < csi_response.csi(csi_i).rx_nodes(rx_i).csi_imag().size() - 1)
                    ss << ",";
            }
            ss << "]" << std::endl;
            //NS_LOG_INFO("    -> CSI " << " (TxId: " << txId << " -> " << rxId << ")" << ss.str());

            // Add the info from all other receivers to the cache
            CacheKey otherkey = CacheKey(txId, rxId);
            // AZU: todo: add CSI to cache
            CacheEntry entry = CacheEntry(delay, wb_loss, start_time, end_time);

            auto cache_it = m_cache.find(otherkey);

            if (cache_it != m_cache.end()) {
                // pair already known
                cache_it->second.push_back(entry);
            } else {
                // create new entry with zero vector
                m_cache.insert(std::make_pair(otherkey, std::vector<CacheEntry>()));
                cache_it = m_cache.find(otherkey);
                cache_it->second.push_back(entry);
            }
        }
    }

    // get result from cache
    CacheKey key2 = CacheKey(node_a->GetId(), node_b->GetId());
    auto cache_it = m_cache.find(key2);
    if (cache_it != m_cache.end())
    {
        // iterate over all stored entries for that link
        for (CacheEntry c_entry : cache_it->second) {
            // If delay and loss exist in the cache, check if the entry is not outdated
            if (c_entry.m_end_time >= current_time && c_entry.m_start_time <= current_time)
            {
                // Return cache entry as the value is still fresh
                return c_entry;
            }
        }
    }
    // cannot be reached
    CacheEntry dummy_entry = CacheEntry(current_time, -1, current_time, current_time);
    return dummy_entry;
}

} // namespace ns3
