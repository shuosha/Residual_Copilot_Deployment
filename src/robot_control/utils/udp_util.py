import socket
import numpy as np
import pickle
import threading
import time

from robot_control.control.common.communication import *

np.set_printoptions(precision=3, suppress=True)

class udpSender:
    def __init__(self, port: dict):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send(self, data: dict):
        self.sock.sendto(pickle.dumps(data), ('127.0.0.1', self.port))

    def close(self):
        self.sock.close()

class udpReceiver:
    def __init__(self, ports: dict, re_use_address = False):
        self.ports = ports
        self.re_use_address = re_use_address
        self.results = {}
        self.lock = threading.Lock()
        self.thread = None

    def receive(self, name, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1)  # Set timeout to 1 second
        if self.re_use_address:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))
        while self.alive:
            try:
                t0 = time.time()
                try:
                    data, addr = sock.recvfrom(32768)
                except socket.timeout:
                    continue
                data = pickle.loads(data)
                with self.lock:
                    self.results[name] = data
                t1 = time.time()
                # print(f"Receive {name} data from {addr} in {t1-t0:.6f} seconds")
            except BaseException as e:
                print(f"Error in receiving {name} data: {e}")
                break
        sock.close()
        print(f"Receive {name} thread stopped")

    def start(self):
        self.alive = True
        self.thread = []
        for name, port in self.ports.items():
            t = threading.Thread(target=self.receive, args=(name, port))
            t.start()
            self.thread.append(t)

    def stop(self):
        print("Stopping udpReceiver")
        self.alive = False
        for t in self.thread:
            if t is not None and t.is_alive():
                t.join()
        print("udpReceiver stopped")

    def get(self, name=None, pop=False):
        with self.lock:
            if name is None:
                ret = self.results
                if pop:
                    self.results = {}
                return ret
            else:
                if name in self.results:
                    if pop:
                        ret = self.results[name]
                        del self.results[name]
                        return ret
                else:
                    return None

if __name__ == '__main__':
    udpReceiver = udpReceiver(ports=
                            {
                            'object_pose': OBJECT_POSE_PORT,
                            'xarm_state': XARM_STATE_PORT
                            })
    udpReceiver.start()
    try:
        while True:
            print(udpReceiver.get())
            print()
            time.sleep(0.01)
    except KeyboardInterrupt:
        try:
            udpReceiver.lock.release()
        except:
            pass
        udpReceiver.stop()
