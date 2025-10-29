#!/bin/bash

# run crowdsource_catalogs_long.py
python crowdsource_catalogs_long.py
# run saturated_star_finding.py
python saturated_star_finding.py
# run update_sat_stars_catalog.py
python update_sat_stars_catalog.py
# run append_sat_cat_to_original_cat.ipynb
jupyter nbconvert --to notebook --execute append_sat_cat_to_original_cat.ipynb --

# run merge_catalog_after_appending_sat_stars 
python merge_catalog_after_appending_sat_stars.py

# run refine_f140m_catalog.ipynb
jupyter nbconvert --to notebook --execute refine_f140m_catalog.ipynb --output
# run refine_f480m_catalog.ipynb
jupyter nbconvert --to notebook --execute refine_f480m_catalog.ipynb --output
# run merge_catalog.ipynb
jupyter nbconvert --to notebook --execute merge_catalog.ipynb --output
#--> will give the complete catalog of nircam and miri

#will make code merging nircam and miri
