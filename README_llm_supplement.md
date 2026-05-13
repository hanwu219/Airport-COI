# LLM Supplement Script

`llm_write_supplement.py` is a small helper script for supplementing the `need_text` field in an Excel file. It reads the first four columns of each row as context: `code`, `description`, `stakeholder`, and `system`, then calls the configured DeepSeek-compatible API to generate a clearer operation requirement text.

The script only writes back the `need_text` column. If the Excel file already contains a `need_text` column, that column is updated. If it does not exist, the script creates it. All other columns are preserved.

Before running, set `DEEPSEEK_API_KEY` in a local `.env` file or in your environment. Then run:

```bash
python llm_write_supplement.py --input-xlsx input.xlsx --output-csv output.csv
```

The default model is `deepseek-v4-flash`, and the default API base URL is `https://api.deepseek.com`.
