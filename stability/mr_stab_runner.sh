# This script is responsible for running stability finders for various orders
# and step ratios of IMEX Adams MR.

# First, construct the PURE EXPLICIT stability regions.
for o in `seq 1 4`;
do

echo "Running pure explicit stability region finder for order $o"
sed -i "s/.*order = .*/    order = $o/" mr_stability_adams.py
sed -i "s/.*rs_mu = .*/    rs_mu = [0]/" mr_stability_adams.py
sed -i "s/.*imex = .*/    imex = False/" mr_stability_adams.py
python mr_stability_adams.py

done

# Then, construct the EXPLICIT IMEX stability regions.
# For now, for each order we attempt to set r_mu at or near
# the implicit stability bound, if one exists.
for o in `seq 1 3`;
do

echo "Running IMEX explicit stability region finder for order $o"
sed -i "s/.*order = .*/    order = $o/" mr_stability_adams.py
sed -i "s/.*rs_mu = .*/    rs_mu = [5]/" mr_stability_adams.py
sed -i "s/.*imex = .*/    imex = True/" mr_stability_adams.py
python mr_stability_adams.py

done

# Let fourth order dangle - it requires a special bound.
echo "Running IMEX explicit stability region finder for order 4"
sed -i "s/.*order = .*/    order = 4/" mr_stability_adams.py
sed -i "s/.*rs_mu = .*/    rs_mu = [2]/" mr_stability_adams.py
sed -i "s/.*imex = .*/    imex = True/" mr_stability_adams.py
python mr_stability_adams.py

# Now, reverse the process, constructing the PURE IMPLICIT stability
# regions for orders where it makes sense to do so (first and second
# order are A-stable and would/will run infinitely).
for o in `seq 3 4`;
do

echo "Running pure implicit stability region finder for order $o"
sed -i "s/.*order = .*/    order = $o/" mr_imp_stability_adams.py
sed -i "s/.*rs_lbda = .*/    rs_lbda = [0]/" mr_imp_stability_adams.py
sed -i "s/.*imex = .*/    imex = False/" mr_imp_stability_adams.py
python mr_imp_stability_adams.py

done

echo "Running IMEX implicit stability region finder for order 3"
sed -i "s/.*order = .*/    order = $o/" mr_imp_stability_adams.py
sed -i "s/.*rs_lbda = .*/    rs_lbda = [0.5]/" mr_imp_stability_adams.py
sed -i "s/.*imex = .*/    imex = True/" mr_imp_stability_adams.py
python mr_imp_stability_adams.py

echo "Running IMEX implicit stability region finder for order 4"
sed -i "s/.*order = .*/    order = $o/" mr_imp_stability_adams.py
sed -i "s/.*rs_lbda = .*/    rs_lbda = [0.3]/" mr_imp_stability_adams.py
sed -i "s/.*imex = .*/    imex = True/" mr_imp_stability_adams.py
python mr_imp_stability_adams.py



