"""
    Sample script showing how to submit a prediction json file to the AWS bucket assigned to you for the challenge.
    Credentials are in your sign-up e-mail: please refer to the full project README for the exact format of the file
    and the naming convention you need to respect.
    Make sure to duplicate the .env.local file as an .env file in this folder, and fill it with the right values
    (or alternatively, set up the corresponding env variables).
    Required packages can be found in the requirements.txt file in this folder.
"""

from datetime import datetime
import os
from secrets import AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET_NAME, EMAIL, PARTICIPANT_ID

import boto3


def upload_submission(local_file: str, task: str):
    """
    Thanks to Alex Egg for catching the bug!
    :param local_file: local path, may be only the file name or a full path
    :param task: rec or cart
    :return:
    """

    print("Starting submission at {}...\n".format(datetime.utcnow()))
    # instantiate boto3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name="us-west-2",
    )
    s3_file_name = os.path.basename(local_file)
    # prepare s3 path according to the spec
    s3_file_path = "{}/{}/{}".format(
        task, PARTICIPANT_ID, s3_file_name
    )  # it needs to be like e.g. "rec/id/*.json"
    # upload file
    s3_client.upload_file(local_file, BUCKET_NAME, s3_file_path)
    # say bye
    print("\nAll done at {}: see you, space cowboy!".format(datetime.utcnow()))

    return


if __name__ == "__main__":
    # LOCAL_FILE needs to be a json file with the format email_epoch time in ms - email should replace @ with _
    pass
    # LOCAL_FILE = '{}_1622483929123.json'#'{}_1616887274000.json'.format(EMAIL.replace('@', '_'))
    # TASK = 'rec'  # 'rec' or 'cart'
    # upload_submission(local_file=LOCAL_FILE, task=TASK)
