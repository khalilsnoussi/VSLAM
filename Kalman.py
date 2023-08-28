import numpy as np




class Kalman:
    def __init__(self) -> None:
        return
    
    def predict(self, x, P, F, Q):      
        x = np.dot(F, x)
        P = np.dot(np.dot(F, P), F.T) + Q
        return x, P
    
    def update(self, x, P, z, H, R):
        y = z - np.dot(H, x)
        S = np.dot(np.dot(H, P), H.T) + R
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        x = x + np.dot(K, y)
        P = P - np.dot(np.dot(K, H), P)
        return x, P
    
    
    
    