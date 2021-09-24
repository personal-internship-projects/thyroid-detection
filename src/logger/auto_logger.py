import inspect, logging

def autolog(message, warn = 1):
    '''
    Automatically log the current function details.
    '''

    func = inspect.currentframe().f_back.f_code
    file_name     = func.co_filename
    function_name = func.co_name
    line_number   = str(func.co_firstlineno)

    if warn == 1:
        logging.info(file_name + " > " + function_name +" - " + message)
    elif warn == 2:
        logging.warn(file_name + " > " + function_name +" - " + message)
    elif warn == 3:
        logging.error(file_name + " > " + function_name +" - " + message)

