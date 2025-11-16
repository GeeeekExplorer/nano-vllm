class Observer():
    ttft_start = 0
    tpot_start = 0

    ttft = 0
    tpot = 0
    
    @classmethod
    def reset_ttft(cls):
        cls.ttft_start = 0
    
    @classmethod
    def complete_reset(cls):
        cls.ttft_start = 0
        cls.tpot_start = 0
        cls.ttft = 0
        cls.tpot = 0