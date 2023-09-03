class KalmanFilterBase:

    def __init__(self):
        self.state = None
        self.covariance = None
        self.innovation = None
        self.innovation_covariance = None

    def get_current_state(self):
        if self.state is not None:
            return self.state.tolist()
        return None

    def get_current_covariance(self):
        if self.covariance is not None:
            return self.covariance.tolist()
        return None

    def get_last_innovation(self):
        if self.innovation is not None:
            return self.innovation.tolist()
        return None

    def get_last_innovation_covariance(self):
        if self.innovation_covariance is not None:
            return self.innovation_covariance.tolist()
        return None

