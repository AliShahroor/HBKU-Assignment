# Gemini API Translation Lab

## Overview

This project demonstrates how to connect Python to Google's Gemini API and automate Arabic-to-English translation using Google Colab. It reads Arabic sentences and their topics from Excel, generates contextual translations, and exports the results.

## Features

- Connect to Gemini using an API key stored in Colab Secrets.
- Explore available Gemini models.
- Translate a single Arabic word.
- Upload and read Excel files.
- Translate multiple sentences automatically.
- Use topics to provide translation context.
- Handle API rate limits with delays and retries.
- Save progress and export translated results.

## Files

| File | Description |
| --- | --- |
| [Gemini_Translation_Lab.ipynb](Gemini_Translation_Lab.ipynb) | Colab notebook with setup, optional model exploration and single-word practice, a five-sentence example, and an optional full-file translation workflow. |
| [Arabic_Sentences_Untranslated.xlsx](Arabic_Sentences_Untranslated.xlsx) | Input workbook containing 100 Arabic sentences, sentence IDs, topics, and an empty `English Translation (API)` column. Upload this file to practice. |
| [Translated_Sentences.xlsx](Translated_Sentences.xlsx) | Completed example containing the same 100 sentences with English translations. |

## How to Run

1. Open `Gemini_Translation_Lab.ipynb` in [Google Colab](https://colab.research.google.com/) using **File → Open notebook → GitHub**, or upload the notebook.
2. Create your own Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey).
3. Open the Colab **Secrets** panel (key icon) and save the key under `GEMINI_API_KEY`.
4. Enable notebook access to that secret.
5. Run the notebook cells in order. The setup cell installs the required libraries and retrieves the key securely:

   ```python
   client = genai.Client(
       api_key=userdata.get("GEMINI_API_KEY")
   )
   ```

6. When prompted, upload `Arabic_Sentences_Untranslated.xlsx`.
7. Review the five-sentence example. To translate and download the full dataset, run the optional full-file translation cell, then review the downloaded `Translated_Sentences.xlsx`.

## Notes

- The optional practice cells demonstrate additional functionality.
- The five-sentence example uses a separate copy of the data; the full-file workflow processes the main dataset.
- You can modify the prompts and experiment with other languages.
- Gemini API usage is subject to rate limits and available quotas. Use the model exploration cell to choose a text-generation model available to your account if the configured model is unavailable.
- The full-file workflow skips existing translations, retries rate-limit errors, and saves progress every 10 rows. If requests fail, the exported file may be partially translated; check its completion count.
- Never publish or share API keys.
