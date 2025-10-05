# This  File is the  is the one we use to uplode the folder with  all  the .nc4 to the cloud

import paramiko
import os

local_folder = "/home/gk/Documents/NASA_PROJECT/MERRA2_daily"
remote_folder = "/home/ubuntu/MERRA2_daily"
hostname = "3.84.108.86"
username = "ubuntu"
key_file = "/home/gk/Downloads/Nasa.pem"  # path to your PEM key

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname, username=username, key_filename=key_file)
sftp = ssh.open_sftp()

# Create remote folder if it doesn't exist
def ensure_remote_dir(path):
    """Recursively ensure a remote directory exists."""
    dirs = []
    while len(path) > 1:
        try:
            sftp.stat(path)
            return
        except IOError:
            dirs.append(path)
            path, _ = os.path.split(path)
    for d in reversed(dirs):
        try:
            sftp.mkdir(d)
        except IOError:
            pass

# Upload files recursively, skipping existing ones with same size
for root, dirs, files in os.walk(local_folder):
    remote_path = os.path.join(remote_folder, os.path.relpath(root, local_folder)).replace("\\", "/")
    ensure_remote_dir(remote_path)

    for file in files:
        local_file = os.path.join(root, file)
        remote_file = os.path.join(remote_path, file).replace("\\", "/")

        # Skip if file already exists and size matches
        try:
            remote_attr = sftp.stat(remote_file)
            local_size = os.path.getsize(local_file)
            if remote_attr.st_size == local_size:
                print(f"⏩ Skipped (already exists): {remote_file}")
                continue
        except IOError:
            # File doesn't exist, proceed with upload
            pass

        # Upload file
        sftp.put(local_file, remote_file)
        print(f"✅ Uploaded: {local_file} -> {remote_file}")

sftp.close()
ssh.close()
print("🎉 Upload completed (skipped existing files).")
