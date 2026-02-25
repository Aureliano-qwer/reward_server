curl -X POST http://10.102.97.40:23334/compute_score \
    -H 'Content-Type: application/json' \
    -d '{
      "data_source": "cloud_test",
      "prompt_str": "1+1=?",
      "response_str": "答案是2",
      "ground_truth": "2"
    }'