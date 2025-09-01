import argparse

def get_cmd_parser():
    parser = argparse.ArgumentParser()
    #present
    # parser.add_argument('-s',"--show",type=bool,default=False)
    
    #config file name
    parser.add_argument('-f','--file',type=str,default='config.yaml')
    parser.add_argument('-g','--log',type=str,default='~/log/fedadvan/')
    # parser.add_argument('-bs','--batch_szie',type = int, default = 10)
    # parser.add_argument('-lr','--learning_rate',type=float,default=0.002)
    # parser.add_argument('-cs','--clients',type=int,default=10)
    
    return parser
    
