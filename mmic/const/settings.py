
class Global:
    total_rounds = 200
    current_round = 1
    
    @classmethod
    def iterator(cls):
        while cls.current_round <= cls.total_rounds:
            yield cls.current_round
            cls.current_round += 1