# Put the NCERT Class 10 Science PDFs here

1. Go to https://ncert.nic.in -> Textbooks -> Class X -> Subject: Science.
2. Download the full current edition (the "Download complete book (zip)" link, or each chapter PDF).
3. Unzip into this folder (sub-folders are fine). Then from the project root run:

       python scripts/ingest.py --list      # check chapter names first
       python scripts/ingest.py             # build ./index

If a chapter is named wrongly or skipped, copy `chapters.json.example` to `chapters.json`, fix the
names, and re-run. The PDFs themselves are git-ignored; the generated `./index` folder is committed.
