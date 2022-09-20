import scp
import paramiko
from config import fileServer_config

def scp_images(image_names):
    with paramiko.SSHClient() as ssh:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(fileServer_config["sshIP"], port=fileServer_config["sshPort"], username=fileServer_config["sshUsername"])

        with scp.SCPClient(ssh.get_transport()) as scp_:
            for image_name in image_names:
                scp_.put(image_name, fileServer_config["sshDirectory"])
