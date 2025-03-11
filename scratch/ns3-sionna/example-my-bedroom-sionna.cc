
/**
 *  Author giannis kokkinis 
 *  
 * simple example to understand 
 * 
 *  We have a bedroom with 1 access point and 1 smartphone
 */


// Sionna models
#include "lib/sionna-helper.h"
#include "lib/sionna-mobility-model.h"
#include "lib/sionna-propagation-cache.h"
#include "lib/sionna-propagation-delay-model.h"
#include "lib/sionna-propagation-loss-model.h"

// Ns-3 modules
#include "../../src/wifi/model/yans-wifi-phy.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/ssid.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/wifi-net-device.h"

//my include here 
#include "ns3/flow-monitor-module.h"
#include "ns3/wifi-phy.h"
#include "ns3/spectrum-analyzer.h"
#include "ns3/spectrum-value.h"

//These includes is only for debugging 
#include <cmath>
#include <unistd.h>
#include "ns3/random-variable-stream.h"


using namespace ns3;

//count the capacity of packets
int packetCounter=0;

NS_LOG_COMPONENT_DEFINE("myBedroomExample");



double get_center_freq(Ptr<NetDevice> nd)
{
    Ptr<WifiPhy> wp = nd->GetObject<WifiNetDevice>()->GetPhy();

    return wp->GetObject<YansWifiPhy>()->GetFrequency() * 1e6;
}


double get_channel_width(Ptr<NetDevice> nd)
{
    Ptr<WifiPhy> wp = nd->GetObject<WifiNetDevice>()->GetPhy();
    return wp->GetObject<YansWifiPhy>()->GetChannelWidth() * 1e6;
}


//create my functions here 



/**
 * 
 *  need to send data to sionna_server.py
 * 
 */
void print_channel_info(double signal_power, Time last_rx_start_time, Time last_rx_end_time)
{
    std::cout <<"---------- CHANNEL INFO -------\n" << std::endl;
    std::cout << "Signal power: " << signal_power << " dBm" << std::endl;
    std::cout << "Last RX start time: " << last_rx_start_time.GetSeconds() << " s" << std::endl;
    std::cout << "Last RX end time: " << last_rx_end_time.GetSeconds() << " s" << std::endl;
    std::cout << "Duration : " << (last_rx_end_time.GetSeconds() - last_rx_start_time.GetSeconds()) << " s" << std::endl;
    std::cout <<"-------------------------------\n" << std::endl;
    //to run always
    Simulator::Schedule(Seconds(2.0), &print_channel_info, signal_power, last_rx_start_time, last_rx_end_time);

}



/**This method called when a server receives a packet... */
void OnReceivePacket(Ptr<const Packet> packet, const Address &address){
    //log info about received packet
    uint32_t size = packet->GetSize(); //get the size of the packet to print
    uint8_t* buffer = new uint8_t[size]; // size count by Bytes
    uint32_t data = packet->CopyData(buffer, size);

    std::cout << "\n----\n " << std::endl;
    std::cout << "Receive ---- packet size: " << size << "Bytes\n" << std::endl;
    std::cout << "Receive ---- packet data: " << data << std::endl;
    std::cout << "\n----\n" << std::endl;

    //test
    std::cout << "Buffer { ";
    for (uint32_t i=0; i<size; i++){
        std::cout << static_cast<int>(buffer[i])  << " , ";
    }
    std::cout << "}\n" << std::endl;
    //end test

    sleep(2);

    packetCounter++;
}





/**
 *  This method here is about check errors inside the channel
 *
 */
void checkChannelErros(){

}

int main(int argc, char* argv[])
{
    bool verbose = true;
    bool tracing = true;
    bool caching = true;
    std::string environment = "simple_room/simple_room.xml";
    int wifi_channel_num = 6;
    int channel_width = 20; // 802.11g supports only 20MHz

    CommandLine cmd(__FILE__);
    cmd.AddValue("verbose", "Enable logging", verbose);
    cmd.AddValue("tracing", "Enable pcap tracing", tracing);
    cmd.AddValue("caching", "Enable caching of propagation delay and loss", caching);
    cmd.AddValue("environment", "Xml file of environment", environment);
    cmd.AddValue("channel", "The WiFi channel number", wifi_channel_num);
    cmd.Parse(argc, argv);

    if (verbose)
    {
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
        LogComponentEnable("YansWifiChannel", LOG_DEBUG);
        LogComponentEnable("YansWifiChannel", LOG_PREFIX_TIME);
        LogComponentEnable("SionnaPropagationDelayModel", LOG_INFO);
        LogComponentEnable("SionnaPropagationCache", LOG_INFO);
        // LogComponentEnable("WifiPhy", LOG_LEVEL_DEBUG);
    }

    std::cout << "Example scenario with sionna" << std::endl << std::endl;

    SionnaHelper sionnaHelper(environment, "tcp://localhost:5555");

    /**
     * 
     *  1 SMARTPHONES
     */
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(1);

    /***
     * 
     *  1 Access Point
     */
    NodeContainer wifiApNode;
    wifiApNode.Create(1); // 1 Access Point

    // Create a channel
    Ptr<YansWifiChannel> channel = CreateObject<YansWifiChannel>();

    //Create cache to reduce repeated calculation
    Ptr<SionnaPropagationCache> propagationCache = CreateObject<SionnaPropagationCache>();
    propagationCache->SetSionnaHelper(sionnaHelper);
    propagationCache->SetCaching(caching);

    //create delay Model
    Ptr<SionnaPropagationDelayModel> delayModel = CreateObject<SionnaPropagationDelayModel>();
    delayModel->SetPropagationCache(propagationCache);

    //create loss model
    Ptr<SionnaPropagationLossModel> lossModel = CreateObject<SionnaPropagationLossModel>();
    lossModel->SetPropagationCache(propagationCache);

    //set the loss and delay model to channel
    channel->SetPropagationLossModel(lossModel);
    channel->SetPropagationDelayModel(delayModel);
    
    

    // WiFi configuration create wifi channel using other library
    YansWifiPhyHelper phy; 
    phy.SetChannel(channel);

   


    

    

    
    //WIFI configurations
    WifiHelper wifi;
    WifiStandard wifi_standard = WIFI_STANDARD_80211g;
    wifi.SetStandard(wifi_standard);

    //set channel stats
    std::string channelStr = "{" + std::to_string(wifi_channel_num) + ", " + std::to_string(channel_width) + ", BAND_2_4GHZ, 0}";
    phy.Set("ChannelSettings", StringValue(channelStr));
    
    //MAC configurations
    WifiMacHelper mac;
    Ssid ssid = Ssid("ns-3-ssid");

    /**configure sta devices */
    NetDeviceContainer staDevices;
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    staDevices = wifi.Install(phy, mac, wifiStaNodes);
    
    /**configure ap devices */
    NetDeviceContainer apDevices;
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid), "BeaconGeneration", BooleanValue(true), "BeaconInterval", TimeValue(Seconds(5.120)), "EnableBeaconJitter", BooleanValue(false));
    apDevices = wifi.Install(phy, mac, wifiApNode);
    
    //Debug MTU of AP and STA
    std::cout << "---- \n --  AP MTU: " << apDevices.Get(0)->GetMtu() << "\n-----\n" << std::endl;
    std::cout << "---- \n --  STA MTU: " << staDevices.Get(0)->GetMtu() << "\n-----\n" << std::endl;


    //collecting informations about physical layer 
    Ptr<WifiPhy> phyInfo = apDevices.Get(0)->GetObject<WifiNetDevice>()->GetPhy();
    //get informations here 
    double signal_power = phyInfo->GetPowerDbm(0);
    Time last_rx_start_time = phyInfo->GetLastRxStartTime();
    Time last_rx_end_time = phyInfo->GetLastRxEndTime();
    
    


    


    // Mobility configuration
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::SionnaMobilityModel");
    mobility.Install(wifiStaNodes);
    mobility.Install(wifiApNode);

    //SET POSITIONS HERE 
    wifiStaNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(5.0, 2.05, 1.0));
    wifiApNode.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(1.0, 2.0, 1.0));

    // Set up Internet stack and assign IP addresses
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);

    //address assignment to devices
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0"); //(NETWORK ADDRESS, MASK)
    Ipv4InterfaceContainer wifiStaInterfaces = address.Assign(staDevices);
    Ipv4InterfaceContainer wifiApInterfaces = address.Assign(apDevices);

    // UDP CONNECTION HERE
    // UdpEchoServerHelper echoServer(9);

    // TCP SERVER
    Address TCPSinkAddress(InetSocketAddress(Ipv4Address::GetAny(),9)); //Create TCP connection and set on port 9 for TCP TRANSFER - Server 
    PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", TCPSinkAddress);
    ApplicationContainer serverApps = packetSinkHelper.Install(wifiApNode);
    serverApps.Start(Seconds(1.0)); //start server 
    serverApps.Stop(Seconds(10.0)); //stop server 



    /**
     * 
     * Creating here the sink to gather the packets Application Layer
     */
    Ptr<PacketSink> sink = serverApps.Get(0)->GetObject<PacketSink>();
    sink->TraceConnectWithoutContext("Rx", MakeCallback(&OnReceivePacket)); //connect with our function



    //TCP CLIENT
    OnOffHelper TCPclient("ns3::TcpSocketFactory", Address(InetSocketAddress(wifiApInterfaces.GetAddress(0), 9))); //Create TCP connection and set on port 9 TCP TRANSFER - Client
    TCPclient.SetAttribute("DataRate", StringValue("1Mbps")); //set the speed here 1 Mbps
    TCPclient.SetAttribute("PacketSize", UintegerValue(1024)); //1024 Bytes max size packet
    TCPclient.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]")); 
    TCPclient.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]")); 
    


    ApplicationContainer clientApps = TCPclient.Install(wifiStaNodes);

    Ptr<OnOffApplication> onOffApp = clientApps.Get(0)->GetObject<OnOffApplication>();
    if (onOffApp){

        //here we are creating custom payload 
        uint8_t updatePayload[4] = {0b001100000, 0x00, 0x00, 0x00};

        //and send the packet 
        Ptr<Packet> packet = Create<Packet>(updatePayload, sizeof(updatePayload));

        
        
    }
    


    clientApps.Start(Seconds(2.0)); //start echoClientlient 
    clientApps.Stop(Seconds(10.0)); //stop client

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // set center frequency & bandwidth for Sionna
    sionnaHelper.Configure(get_center_freq(apDevices.Get(0)), get_channel_width(apDevices.Get(0)));

    // Tracing
    if (tracing)
    {
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        phy.EnablePcap("example-sionna", apDevices.Get(0));
        phy.EnablePcap("example-sionna", staDevices.Get(0));
    }

    if (verbose)
    {
        // Print node information
        std::cout << "----------Node Information----------" << std::endl;
        NodeContainer c = NodeContainer::GetGlobal();
        for (auto iter = c.Begin(); iter != c.End(); ++iter)
        {
            std::cout << "NodeID: " << (*iter)->GetId() << ", ";

            Ptr<MobilityModel> mobilityModel = (*iter)->GetObject<MobilityModel>();
            if (mobilityModel)
            {
                // Tracing
                if (tracing)
                {
                    phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
                    phy.EnablePcap("example-sionna", apDevices.Get(0));
                    phy.EnablePcap("example-sionna", staDevices.Get(0));
                }
            
                std::cout << mobilityModel->GetInstanceTypeId().GetName() << " (";
                Vector position = mobilityModel->GetPosition();
                Vector velocity = mobilityModel->GetVelocity();
                std::cout << "Pos: [" << position.x << ", " << position.y << ", " << position.z << "]" << ", ";
                std::cout << "Vel: [" << velocity.x << ", " << velocity.y << ", " << velocity.z << "]";
                
                Ptr<SionnaMobilityModel> sionnaMobilityModel = DynamicCast<SionnaMobilityModel>(mobilityModel);
                if (sionnaMobilityModel)
                {
                    std::cout << ", " << "Model: " << sionnaMobilityModel->GetModel() << ", ";
                    std::cout << "Mode: " << sionnaMobilityModel->GetMode() << ", ";
                    std::cout << "ModeTime: " << sionnaMobilityModel->GetModeTime().GetSeconds() << ", ";
                    std::cout << "ModeDistance: " << sionnaMobilityModel->GetModeDistance() << ", ";
                    std::cout << "Speed: " << sionnaMobilityModel->GetSpeed()->GetInstanceTypeId().GetName() << ", ";
                    std::cout << "Direction: " << sionnaMobilityModel->GetDirection()->GetInstanceTypeId().GetName();
                }
                std::cout << ")" << std::endl;
            }
            else
            {
                std::cout << "No MobilityModel" << std::endl;
            }
        }
    }

    //create here monitor 
    FlowMonitorHelper flow;
    Ptr<FlowMonitor> monitor = flow.InstallAll();



    Simulator::Stop(Seconds(1000.0));

    sionnaHelper.Start();


    // print channel state info
    Simulator::Schedule(Seconds(2.0), &print_channel_info, signal_power, last_rx_start_time, last_rx_end_time);

    Simulator::Run();

    

    //export stats to results
    //monitor->SerializeToXmlFile("results.xml", true, true);

    Simulator::Destroy();

    

    //print the packets here 
    uint64_t totalBytesReceived = sink->GetTotalRx();
    std::cout << "Total Received: " << packetCounter << " packets" << std::endl;
    std::cout << "Total Received: " << totalBytesReceived << " Bytes" << std::endl;

    std::cout << "Ns3-sionna: cache hit ratio: " <<  propagationCache->GetStats() << std::endl;

    sionnaHelper.Destroy();

    return 0;
}
