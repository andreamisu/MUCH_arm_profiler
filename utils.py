import getpass
import subprocess

def getSudoPermissions():
    #deal with sudo permissions while running a normal-user bash script
    sudo_password = getpass.getpass(prompt='sudo password: ')
    p = subprocess.Popen(['sudo', '-S', 'ls'], stderr=subprocess.PIPE, stdout=subprocess.PIPE,  stdin=subprocess.PIPE)
    try:
        out, err = p.communicate(input=(sudo_password+'\n').encode(),timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
        return False
    return True