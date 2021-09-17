

# Submission

To create a submission for the next-item recommendation task, you should run the foollowing command:


`python3 generate_subsmission.py ARGS1 ARGS2 ...`


Arguments:

- `--train_session_file` - path to browsing logs

- `--test_file1` - path to the test file of the first stage


- `--test_file2` - path to the test file of the second stage


- `--predictor_path` - path to url2sku heuristic

- `--top_k_sessions` - num of rows to read from browsing train

- `--email`, `--bucket_name`, `--participant_id`, 
`--aws_access_key`, `--aws_secret_key` - params for submission

- `--upload_submission` - whether we want to submit our predictions

- `--file` - test file for inference
