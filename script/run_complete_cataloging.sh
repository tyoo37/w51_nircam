#!/bin/bash

# run crowdsource_catalogs_long.py
# run daophot using photutils
python crowdsource_catalogs_long.py

# run saturated_star_finding.py
# this will make saturated star list but it contains stars with bad pixels as well
python saturated_star_finding.py

# run update_sat_stars_catalog.py
# so running psf photomtery again for cutout made by each saturated star. especially bad centering is fixed.
python update_sat_stars_catalog.py

# run append_sat_cat_to_original_cat.ipynb
# append the updated saturated star catalog to original catalog. prioritize the saturated star when they are merged by spatial matching.
jupyter nbconvert --to notebook --execute append_sat_cat_to_original_cat.ipynb --

# run merge_catalog_after_appending_sat_stars 
# merge across exposures and across filters to make the complete catalog including saturated stars
python merge_catalog_after_appending_sat_stars.py

# run refine_f140m_catalog.ipynb
# start refining process from f140m and f480m catalog -- most of stars will be detected in at least one of the two filters. 
# compare paramters space including qfit, cfit, sharpness, roundness1, roundness2, snr, etc to filter out bad sources.
jupyter nbconvert --to notebook --execute refine_f140m_catalog_after_sat.ipynb --output
# run refine_f480m_catalog.ipynb
jupyter nbconvert --to notebook --execute refine_f480m_catalog_after_sat.ipynb --output
# run merge_catalog.ipynb
# merge f140m and f480m refined catalog to make the final nircam catalog. The other filter catalogs are incorporated during the merging process.
jupyter nbconvert --to notebook --execute merge_catalog_f140m_f480m_after_sat.ipynb --output
#--> will give the complete catalog of nircam and miri

#will make code merging nircam and miri
