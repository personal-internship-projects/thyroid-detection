from os.path import isdir 
from os import system, getcwd
import logging

if isdir("src/logs"):
    with open("src/logs/scraper.log",'w'):
        pass
    log_path = "src/logs/scraper.log"

else:
    with open(f"{getcwd()}/scraper.log",'w'):
        pass
    log_path = f"{getcwd()}/scraper.log"

format_var = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.basicConfig(filename=log_path, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG)

console_log = logging.StreamHandler()
console_log.setLevel(logging.INFO)
console_log.setFormatter(format_var)
logging.getLogger("").addHandler(console_log)
