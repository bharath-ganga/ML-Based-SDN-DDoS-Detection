from mininet.net import Mininet
from mininet.node import Controller
from mininet.topo import TreeTopo

topo = TreeTopo(depth=2, fanout=2)
net = Mininet(topo=topo, controller=Controller)
net.start()
net.pingAll()
net.stop()
