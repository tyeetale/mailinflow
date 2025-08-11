# mailinflow
simple free ml based email triage system to get to zero inbox

free version of this thing: https://www.getinboxzero.com/


# Instructions (Need to be Simplier)
Create Folders in Your Email Client: Before running the script, go into your email client (like Gmail, Fastmail, etc.) and create the folders the script will use. Based on the default config, you would create:

A main folder called Triage.

Inside Triage, create the sub-folders:

01_Action_Required

02_Travel

03_Finance

04_Personal

05_Newsletters

You will also need an Archive folder at the root level if you don't have one.

Label and Train:

The process is the same, but you now have more categories to choose from. Be thorough. A good dataset is key.

python email_triage.py --label

python email_triage.py --train

Run and Automate:

First, test with a dry run: python email_triage.py --run --dry-run

See how it classifies your current unread emails. It should tell you where everything would go.

When ready, unleash it: python email_triage.py --run

Your inbox will be cleared, and all unread emails will be neatly filed in your new triage folders, ready for your review.