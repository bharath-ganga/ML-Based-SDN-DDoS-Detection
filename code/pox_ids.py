from pox.core import core
from pox.openflow import libopenflow_01 as of
import joblib
import numpy as np

log = core.getLogger()
model = joblib.load("models/RandomForest.pkl")

def _handle_PacketIn(event):
    flow_features = np.random.rand(1,7)  # placeholder for real flow stats
    prediction = model.predict(flow_features)

    if prediction[0] == 1:
        log.warning("⚠️  DDoS Attack Detected")
    else:
        log.info("Normal Traffic")

def launch():
    core.openflow.addListenerByName("PacketIn", _handle_PacketIn)
    log.info("SDN IDS Module Running")
