#!/usr/bin/env python
# ------------------------------------------------------------------------------------
# Name: eso_download_raw_and_calibs_2025-12-02.py
# Version: 2025-12-02 
# Changes: Many! First were: removed .decode()'s added fitsverify as per ESO User Support
# Version_original: 2019-11-21
# Author: David Petit
# Author_original: A.Micol, Archive Science Group, ESO
# Purpose: Python 3 example on how to download raw science frames
#          and, for each of them, the associated calibration files.
#!!!!!!!!
#!CAVEAT! This is just an example showing the business logic.
#!CAVEAT! There is:
#!CAVEAT!  - little error handling in place.
#!CAVEAT!  - little optimization regarding files downloaded
#!CAVEAT!  - no check if enough disk space is available for download.
#!CAVEAT!  - etc.
#!CAVEAT! Use at your own risk.
#!!!!!!!!
#
# (Original) business logic:
# - FIND SOME SCIENCE RAW FILES
# - FOR EACH SCIENCE RAW FILE:
#     1.- RETRIEVE IT
#     2.- GET ITS LIST OF CALIBRATION FILES (without saving it into a file)
#     3.- PRINT CASCADE INFORMATION AND MAIN DESCRIPTION
#     4.- DOWNLOAD EACH #calibration FILE FROM THE LIST
#     5.- PRINT ANY ALERT OR WARNING ENCOUNTERED BY THE PROCESS THAT GENERATES THE CALIBRATION CASCADE
#
# Limitation: This script does not download siblings, nor the association trees.
#
# Documentation: http://archive.eso.org/cms/application_support/calselectorInfo.html
# 
# Contact: In case of questions, please send an email to: usd-help@eso.org 
#          with the following subject: programmatic access (eso_download_raw_and_calibs.py)
# -------
# The script will get an astropy WARNING (a meaningless solved bug in their code):
# WARNING: W35: None:6:6: W35: 'value' attribute required for INFO elements [astropy.io.votable.tree]
# https://github.com/astropy/astropy/issues/9646
# ------------------------------------------------------------------------------------
import os
import sys
import math
import pyvo
from pyvo.dal import tap
import requests
import cgi
import re
from astropy.coordinates import SkyCoord # Celestial coordinate handling and conversions
import subprocess
import time
from datetime import datetime, timedelta
import pandas as pd

# Decide what you want to download:
mode_requested = "raw2raw"  # other choice:raw2master 

# Set a User Agent (modify as you like, but please let intact the python version used for our usage statistics):
thisscriptname = os.path.basename(__file__)
headers = {}
headers={'User-Agent': '%s (ESO script drc %s)'%(requests.utils.default_headers()['User-Agent'], thisscriptname)}
# If instead of the requests package, urllib is used:
#headers={'User-Agent': '%s (ESO script drc %s)'%(urllib.request.URLopener.version, thisscriptname)}


def is_fits_file_valid(filename):
    """Use fitsverify to check FITS file integrity.""" # reference: https://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/
    try:
        result = subprocess.run(['fitsverify', filename], capture_output=True, text=True) # dl and 'tar -xvzf fitsverify_v4.20_linux64.tar.gz' 
        return "0 error" in result.stdout.lower()  # crude check; refine as needed
    except Exception as e:
        print(f"Error running fitsverify on {filename}: {e}")
        return False

def download_asset_oldOriginal(url, filename=None):
    response = requests.get(url, stream=True, headers=headers)
    if filename == None:
        contentdisposition = response.headers.get('Content-Disposition')
        if contentdisposition != None:
            value, params = cgi.parse_header(contentdisposition)
            filename = params["filename"]
            # when obsolete, replace with filename = dict([p.split('=') if '=' in p else (p, None) for p in contentdisposition.split(';')[1:]]).get('filename', url.split('/')[-1])
        if filename == None:
            # last chance: get anything after the last '/'
            filename = url[url.rindex('/')+1:]
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=50000):
                f.write(chunk)
    return (response.status_code, filename)

def download_asset(url, filename=None, max_retries=3, retry_delay=2, skip_existing=True):
    response = requests.head(url, headers=headers, allow_redirects=True)
    # Infer filename
    if filename is None:
        contentdisposition = response.headers.get('Content-Disposition')
        if contentdisposition:
            _, params = cgi.parse_header(contentdisposition) # when obsolete, replace with: filename = dict([p.split('=') if '=' in p else (p, None) for p in contentdisposition.split(';')[1:]]).get('filename', url.split('/')[-1])
            filename = params.get("filename", url.split("/")[-1])
        else:
            filename = url.split("/")[-1]
    # Check existance
    if skip_existing and os.path.exists(filename):
        if filename.endswith(".fits") or filename.endswith(".fits.Z"):
            print(f"Checking existing file: {filename}")
            if is_fits_file_valid(filename):
                print(f"Skipping valid existing file: {filename}")
                return 200, filename
            else:
                print(f"File exists but failed fitsverify: {filename}. Re-downloading...")
    # Download with retries
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=(9, 90)) # connect 9s; chunk 90s (read timeout)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=50000):
                        if chunk:
                            f.write(chunk)
                if filename.endswith(".fits") or filename.endswith(".fits.Z"):
                    if is_fits_file_valid(filename):
                        print(f"✅ Successfully downloaded and verified: {filename}. (status: {response.status_code})")
                        return 200, filename
                    else:
                        print(f"❌ Downloaded but FITS verification failed: {filename}. Retrying...")
                        os.remove(filename)
                else:
                    print(f"✅ Successfully downloaded: {filename}")
                    return 200, filename
            else:
                print(f"❌ Download failed (status {response.status_code}) for {filename}")
        except Exception as e:
            print(f"❌ Exception during download of {filename}: {e}")
        
        print(f"🔁 Retrying in {retry_delay} seconds... ({attempt+1}/{max_retries})")
        time.sleep(retry_delay)
    print(f"❌ Failed to download {filename} after {max_retries} attempts.")
    return response.status_code, filename


def calselector_info(description):
    """Parse the main calSelector description, and fetch: category, complete, certified, mode, and messages."""

    category=""
    complete=""
    certified=""
    mode=""
    messages=""
    
    m = re.search('category="([^"]+)"', description)
    if m:
        category=m.group(1)
    m = re.search('complete="([^"]+)"', description)
    if m:
        complete=m.group(1).lower()
    m = re.search('certified="([^"]+)"', description)
    if m:
        certified=m.group(1).lower()
    m = re.search('mode="([^"]+)"', description)
    if m:
        mode=m.group(1).lower()
    m = re.search('messages="([^"]+)"', description)
    if m:
        messages=m.group(1)

    return category, complete, certified, mode, messages

def print_calselector_info(description, mode_requested):
    """Print the most relevant params contained in the main calselector description."""

    category, complete, certified, mode_executed, messages = calselector_info(description)

    alert=""
    if complete!= "true":
        alert = "ALERT: incomplete calibration cascade"

    mode_warning=""
    if mode_executed != mode_requested:
        mode_warning = "WARNING: requested mode (%s) could not be executed" % (mode_requested)

    certified_warning=""
    if certified != "true":
        certified_warning = "WARNING: certified=\"%s\"" %(certified)

    print("    calibration info:")
    print("    ------------------------------------")
    print("    science category=%s" % (category))
    print("    cascade complete=%s" % (complete))
    print("    cascade messages=%s" % (messages))
    print("    cascade certified=%s" % (certified))
    print("    cascade executed mode=%s" % (mode_executed))
    print("    full description: %s" % (description))
   
    return alert, mode_warning, certified_warning



# 28 Nov 2025
def parse_time(timestr):
    """
    Convert a time string from ESO TAP or VOTable to a Python datetime object.

    Handles:
      - ISO format with or without microseconds
      - trailing 'Z' (Zulu / UTC)
      - ESO style 'YYYY-MM-DDTHH:MM:SS.sss' (from VOTable)
      - plain 'YYYY-MM-DD HH:MM:SS' (from TAP)
    """
    if timestr is None:
        return None

    timestr = timestr.rstrip('Z')
    timestr = timestr.replace('T', ' ')

    # Try with microseconds
    try:
        return datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        pass

    # Try without microseconds
    try:
        return datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass

    raise ValueError(f"Cannot parse time string: {timestr}")



# ----------------------------
# TARGET INFORMATION DATABASE
# ----------------------------

def get_target_information(target):
    # Select which star & exoplanet to get the data from
    coords = SkyCoord.from_name(target) # Convert target name to coordinates
    ra = coords.ra.deg # Right Ascension in degrees
    dec = coords.dec.deg # Declination in degrees
    print(f"\tThe target ({target}) has ra and dec of:", ra, dec)
    if target == "Beta Pic":    # works
        date_start = "2020-02-07 12:00:00"
        date_end = "2020-02-08 12:00:00"
    elif target == "PZ TEL":    # This fails during REDUCTION--IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
        date_start = "2019-09-06 12:00:00"
        date_end = "2019-09-07 12:00:00"
    elif target == "51 Eri": # REDUCTION failure first: Later:ValueError: n_comp=2 must < min(n_samples, n_features)=2 w/ ... (even fresh run)
        date_start = "2015-09-24 12:00:00"
        date_end = "2015-09-25 12:00:00"
    elif target == "HR 8799":   # worked the first time, and then the ValueError: n_components=1 must be < min(n_samples, n_features)=1, error...
        date_start = "2021-08-20 12:00:00"
        date_end = "2021-08-21 12:00:00"
    elif target == "HD 203030": # 30 Jul: ValueError: n_comp=2 must < min(n_samp, n_fe)=2 w/... NAXIS1=290, NAXIS2=290, NAXIS3=2, NAXIS4=39
        date_start = "2015-06-23 12:00:00"
        date_end = "2015-06-24 12:00:00"
    elif target == "HIP 78530": # 31 Jul: Works, feint planet undetectable by eye, NAXIS1=290, NAXIS2=290, NAXIS3=3, NAXIS4=39 
        date_start = "2015-05-03 12:00:00"
        date_end = "2015-05-04 12:00:00"
    elif target == "Gliese 504": # 31 Jul: Works, feint planet, maybe NE speckle zone, NAXIS1=290, NAXIS2=290, NAXIS3=12, NAXIS4=39
        date_start = "2020-03-16 12:00:00"
        date_end = "2020-03-17 12:00:00"
    elif target == "HD 95086": # 31 Jul: [   ERROR] coadd_value (2) must be < NDIT (1)
        date_start = "2023-05-01 12:00:00"
        date_end = "2023-05-02 12:00:00"
    elif target == "HIP 65426": # 5 Aug: fails. no OBJECT,CENTER file in the data set. file 1/192: SPHER.2017-05-04T03:02:01.191, DIT #0 [ WARNING] No OBJECT,CENTER file in the dataset. Images will be centered using default center (145,145) Traceback (most recent call last): File "/home/dcpetit/Documents/kuleuven_astronomy/thesis_publication/machineLearning_project/ML_ASDI/lib/python3.11/site-packages/sphere/IFS.py", line 3358, in sph_ifs_combine_data:     cx, cy = centers[wave_idx, :]              ~~~~~~~^^^^^^^^^^^^^ IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
        date_start = "2017-05-03 12:00:00"
        date_end = "2017-05-04 12:00:00"
    elif target == "TYC 8998-760-1": # 5 Aug: ValueError: n_components=2 must be strictly less than min(n_samples, n_features)=2 with svd_solver='arpack' ..
        date_start = "2022-02-13 12:00:00"
        date_end = "2022-02-14 12:00:00"
    elif target == "HD 135344":
        date_start = "2024-03-05 12:00:00"
        date_end = "2024-03-07 12:00:00"
    else:
        print(f"ERROR: The target selected ({target}) doesn't have a start & end date") 
    return coords, ra, dec, date_start, date_end



# --- instantiate the ESO TAP service for raw and processed data:

ESO_TAP_OBS = "http://archive.eso.org/tap_obs"
tapobs = tap.TAPService(ESO_TAP_OBS)

# --- Query TAP for science raw
#print('displaying: tapobs.describe("dbo.raw").columns')
#print(tapobs.describe('dbo.raw').columns)
print()
print("Looking for SCIENCE frames") # belonging to prog_id=0104.C-0418(F) (before: 086.D-0165(A) and '60.A-9255(A) ).\n" )
print("Querying the ESO TAP service at: %s" %(ESO_TAP_OBS))

# Select which star & exoplanet to get the data from
target = "Beta Pic"
coords, ra, dec, date_start, date_end = get_target_information(target)


# From ESO          
query=f"""
    SELECT dp_id
    from dbo.raw
    where instrument = 'SPHERE' and dp_cat='SCIENCE' and dp_tech='IFU'
    and exp_start between '{date_start}' and '{date_end}'
    and intersects(s_region, circle('', {ra}, {dec}, 0.16666667)) = 1
    """ 
    #     SELECT dp_id, exp_start
    #    AND (
    # dp_type LIKE '%OBJECT%' OR dp_type LIKE '%CENTER%' OR dp_type LIKE '%FLUX%' OR
    # tpl_name LIKE '%OBJECT%' OR tpl_name LIKE '%CENTER%' OR tpl_name LIKE '%FLUX%'
    #) 
    #ORDER BY exp_start
    # AND (tpl_type LIKE '%OBJECT%' ... AND (observation_type LIKE '%OBJECT%' ... AND (dpr_type LIKE '%OBJECT%' ... # errored

#   observation     date            program ID          # of fits   object_id   ra      dec         notes
#   Bet Pic         1 May 2023      111.24KM.002        17          --
#   Bet Pic         8 Feb 2020      0104.C-0418(F)      37          --          86.82   -51.0665
#   51 Eri          25 Sep 2015     095.C-0298(D)       47 (23?)    200363269   69.40   -2.47355
#   HR 8799         21 Aug 2021     1104.C-0416(E)      9           --          346.87  21.13425
#   V PZ TEL        2019-09-07      1100.C-0481(P)      11          --          283.27  -50.1805    aka: PZ Tel, HD 174429
#   HD 203030 B     2015-06-24                          9?          --          319.74  26.23054
# 	HIP 78530 b     2015-05-04                          19          --          240.48  -21.9804
#       ''          2018-08-21 (and 2025!)              9
#	Gliese 504 b    2020-03-17 (and many before)        17          --          199.19  9.424156
#	HD 95086 b      2023-05-02 (and many before)        11          --          64.26   -68.6673
#	HIP 65426 b                                         48                      201.15  -51.5045
#	TYC 8998-760-1  2022-02-14                          13          --          201.30  -64.9391    aka YSES 1 
# 	HN Pegasi b     # N/A #
#   HD 135344       2024-03-06                          40                     
#   unknown         ?               60.A-9255(A)        --          --          --      --

print("")
print(query)
print("")


print('\n\tThe variable "query" should be some identifiers (37 for beta pic), and is:')
rawframes = tapobs.search(query=query)
science_rawframes = rawframes
print(rawframes.to_table())
print("")
print("Number of raw frames found:", len(rawframes))
print(rawframes)

print("")


## --- 21 July 2025: "To get them you need to concatenate those 37 ids into the following URL:"
#TheURL = "https://archive.eso.org/calselector/v1/associations?mode=Raw2Master&responseformat=votable"
#for i in range(len(rawframes)):
#    TheURL += "&dp_id=" + str(rawframes[i])[2:-3]
    
# --- 28 Nov 2025: Replace the above TheURL definition:
# Extract dp_id properly
raw_dp_ids = rawframes['dp_id'].astype(str).tolist()
#raw_dp_ids = [str(row['dp_id']) for row in rawframes.to_table()]

TheURL = "https://archive.eso.org/calselector/v1/associations?mode=Raw2Master&responseformat=votable"
print("Start with TheURL =", TheURL)

for dp_id in raw_dp_ids:
    TheURL += "&dp_id=" + dp_id
print("\nAdd the 'for dp_id in raw_dp_ids' to TheURL =", TheURL)



# --- Download the TheURL
from astropy.io.votable import parse_single_table
import io
from datetime import datetime
# Step 1: Get the VOTable from the calselector
print("\nDownloading VOTable from calSelector...")
response = requests.get(TheURL, headers=headers)
if response.status_code != 200:
    print(f"Failed to retrieve VOTable (status code {response.status_code})")
    sys.exit(1)
# Step 2: Parse the VOTable

### Commenting out "votable" variables on 2 Dec 2025, 14:00
#votable_table = parse_single_table(io.BytesIO(response.content)).to_table()
#votable_df = pd.DataFrame(votable_table.as_array())

###votable = parse_single_table(io.BytesIO(response.content)).to_table()

# Step 3: Print basic info
#print(f"Retrieved VOTable with {len(votable)} rows")
#print("Columns in VOTable:", votable.colnames)

# Step 4: Extract download URLs
# ESO often uses 'access_url' or similar for file links
#download_urls = votable_df['access_url']
###download_urls = votable['access_url']
### Step 5: Show calibration links
#print("\nPreview of first few calibration links:")
#for i, row in enumerate(votable[:5]):
#    print(f"{i+1:2d}. {row['eso_category']:20} | {row['description']}")
    
# --- Step 6: Filter to only 32 raw + 3 OBJECT/CENTER frames ---
# --- Separate science/raw frames vs calibration ---
#science_rows = [row for row in votable if row['eso_category'] == 'SCIENCE']
#calib_rows   = [row for row in votable if row['eso_category'] != 'SCIENCE']
#print(f"Total rows in VOTable: {len(votable)}")
#print(f"Science raw frames: {len(science_rows)}")
#print(f"Calibration/extra frames: {len(calib_rows)}")

    
# ============================================================
# 4. DOWNLOAD EACH CALIBRATION FILE FROM YOUR LONG TheURL
# ============================================================

# TheURL is your long string with 37 dp_id= parameters
# Example:
# TheURL = "https://archive.eso.org/...&dp_id=SPHER.2020...&dp_id=..."

# ---------------------------------------
# Convert the single TheURL into individual asset URLs
# ---------------------------------------

parts = TheURL.split("dp_id=")

dp_ids = []
for p in parts[1:]:                       # skip the first chunk (before first dp_id=)
    dp_id = p.split("&")[0].strip()       # isolate dp_id until &
    if dp_id:
        dp_ids.append(dp_id)

print("\nExtracted", len(dp_ids), "dp_id values from TheURL")

# Build calib_urls list in the same structure your code expects:
#       [(url, category), (url, category), ...]
calib_urls = []
for dp in dp_ids[len(science_rawframes):]:
    asset_url = f"https://archive.eso.org/calselector/v1/asset/{dp}"
    calib_urls.append((asset_url, "dp_id"))

try:
    print("Example first URL:", calib_urls[0])
except: 
    print("\nWARNING: 'calib_urls[0]' was unable to be printed... probably an empty variable (or scalar?)\n") 
print("Total URLs prepared:", len(calib_urls))

# ============================================================
# 4b. BEGIN DOWNLOADING LOOP (your original logic)
# ============================================================
# ------------------------------------------------------------
# DOWNLOAD SCIENCE FRAMES (based on dp_ids from TheURL)
# ------------------------------------------------------------

print("\n\t### ### ### BEGIN SCIENCE FRAME DOWNLOADS ### ### ###")
print("\t### Total science dp_ids found:", len(dp_ids[:len(science_rawframes)]), "\n")

n_sci = 0     # counter for science frames
science_urls = []
nfiles = 0
calib_association_trees = []
calselector_url_aggregate = []
calib_urls_aggregate = []
cntr_skipCalSelect = 0
cntr_skipCal = 0

# Build science-frame URLs exactly like your rawframes loop does:
for raw in dp_ids[:len(science_rawframes)]:
    #print("\nthe 'raw' variable is:", raw)
    sci_url = f"http://archive.eso.org/datalink/links?ID=ivo://eso.org/ID?{raw}&eso_download=file"
    #print("sci_url with {raw} is:\n", sci_url)
    #sci_url = f"http://archive.eso.org/datalink/links?ID=ivo://eso.org/ID?{raw['dp_id']}&eso_download=file"
    #print("sci_url with {raw[\"dp_id\"]} is:\n", sci_url)
    science_urls.append(sci_url)

for sci_url in science_urls:
    n_sci += 1
    print(f'\n### SCIENCE FRAME {n_sci}/{len(science_urls)}')
    print("SCI URL:", sci_url)

    try:
        status, sci_filename = download_asset(sci_url)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        status, sci_filename = None, None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error occurred: {e}")
        status, sci_filename = None, None
    except requests.exceptions.Timeout as e:
        print(f"Timeout occurred: {e}")
        status, sci_filename = None, None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        status, sci_filename = None, None
    except Exception as e:
        print(f"WARNING!!!: An unexpected error occurred while downloading the science asset: {e}")
        status, sci_filename = None, None

    if status == 200:
        print("SCIENCE: %4d/%d dp_id: %s downloaded" %
              (n_sci, len(science_urls), sci_filename))
    else:
        print("SCIENCE: %4d/%d dp_id: %s NOT DOWNLOADED (status:%s)" %
              (n_sci, len(science_urls), sci_filename, status))
    print("\n\n")
    # Now download the associated calibration files for this i-th science file
    ###
    calselector_url="http://archive.eso.org/calselector/v1/associations?dp_id=%s&mode=%s&responseformat=votable" % (raw, mode_requested) # .decode()
    #calselector_url="http://archive.eso.org/calselector/v1/associations?dp_id=%s&mode=%s&responseformat=votable" % (raw["dp_id"], mode_requested) # .decode() # This is the original
    print("\ncalselector_url, should be like \nhttp://archive.eso.org/calselector/v1/associations?dp_id=SPHER.2021-05-17T02:38:16.310&mode=raw2raw&responseformat=votable, its\n"+str(calselector_url))
    if calselector_url not in calselector_url_aggregate:
        calselector_url_aggregate.append(calselector_url)
        datalink = pyvo.dal.adhoc.DatalinkResults.from_result_url(calselector_url)
        skip_csurl = 0
        print("\ncalselector_url is not in calselector_url_aggregate, so append it in, skip_csurl = 0; and\n'datalink' (pyvo.dal.adhoc.DatalinkResults.from_result_url(calselector_url)) is a table of links with a length of: ", len(datalink))
    else:
        print("calselector_url is already in calselector_url_aggregate, so skip_csurl = 1; and skip the download of:\n"+str(calselector_url)+".")
        cntr_skipCalSelect += 1
        skip_csurl = 1
    #   datalink = pyvo.dal.adhoc.DatalinkResults.from_result_url(calselector_url)

    #  ... 3.- PRINT CASCADE INFORMATION AND MAIN DESCRIPTION
    # ---------------------------------------
    print("\nCascade info & description are...")
    this_description=next(datalink.bysemantics('#this')).description 
    alert, mode_warning, certified_warning = print_calselector_info(this_description, mode_requested)

    # create and use a mask to get only the #calibration entries:
    calibrators = datalink['semantics'] == '#calibration'
    calib_urls = datalink.to_table()[calibrators]['access_url','eso_category']
    # urls_only = set(calib_urls['access_url']) # and then set the "calib_urls" in the next line to this "urls_only"
    # break calib_urls down into parts (with an index number). for each part, if it's a repeat, delete that part (by its index #)
    print_out_info = 0
    if print_out_info == 0:
        print('\n\nstr(calib_urls) is:\n', str(calib_urls)) # The long list of URLs with category, seemingly
        print('\n\nstr(calib_urls[3]) is:\n', str(calib_urls[3])) # This is the 3rd URL with category, seemingly
        print('\n\nstr(calib_urls[0][:]) is:\n', str(calib_urls[0][:])) # This is the long list of URLs (without category), seemingly
        print('\n\nstr(calib_urls[:][0]) is:\n', str(calib_urls[:][0])) # This is the 0th url with category, seemingly
        print('\n\nstr(calib_urls[3][0]) is:\n', str(calib_urls[3][0])) # This is 3rd of long list of URLs! 
        print('\n\nstr(calib_urls["access_url"]) is:\n', str(calib_urls["access_url"])) # This is...
        print("Check if each URL in calib_urls (which is:\n"+str(calib_urls)+") is already in the aggregate calib_urls, if not, make it to DL")
    print("Check if each URL in calib_urls is already in the aggregate calib_urls, if not, make it to DL\n")
    calib_urls_thisURL = []
    for i in range(len(calib_urls["access_url"])): # Seems wrong, fix? -- Double check this... maybe np.shape(calib_urls)[0] or something, but numpy isn't installed
        if str(calib_urls["access_url"][i]) not in calib_urls_aggregate: # i-th calib_urls not in c_urls_aggregate: 
        #if str(calib_urls[i][0]) not in calib_urls_aggregate: # i-th calib_urls not in c_urls_aggregate: 
            print('str(calib_urls['+str(i)+'][0]) is:\n', str(calib_urls[i][0])) # This is 3rd of long list of URLs! 
            calib_urls_aggregate.append(calib_urls[i][0]) #add just the url to the DB
            calib_urls_thisURL.append(calib_urls[i]) #add the url and the category to the download list
            skip_curl = 0
            #print("\tstr(calib_urls[i][0]), which is: "+str(calib_urls[i][0])+", isn't in calib_urls_aggregate, append it, make skip_csurl = 0")
        else:
            # After the download script works well, delete or comment out the print statement below
            print("str(calib_urls["+str(i)+"][0]) is already in calib_urls_aggregate, skip the (i-th) download of:\n\t"+str(calib_urls[i][0])+".")
            cntr_skipCal += 1
            skip_curl = 1
    
    # ------------------------------------------------------------
    # DOWNLOAD Calibration FRAMES
    # ------------------------------------------------------------
    #  ... 4.- DOWNLOAD EACH #calibration FILE FROM THE LIST that is unique and/or appearing for the first time
    # ---------------------------------------
    calib_urls = calib_urls_thisURL
    if True: # skip_csurl == 0 and skip_curl == 0:
        print('\n\t### ### ### The files in calib_urls will be downloaded, their length is: ', len(calib_urls))
        i_calib=0
        print('\n\t### Begin the (original) downloading calibrations for loop...')
        for url,category in calib_urls:
            i_calib+=1
            print('\t### The download_asset(url)\'s category & url are:', category, url)
            try:
                status, filename = download_asset(url)
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error occurred: {e}")
                status, filename = None, None
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error occurred: {e}")
                status, filename = None, None
            except requests.exceptions.Timeout as e:
                print(f"Timeout occurred: {e}")
                status, filename = None, None
            except FileNotFoundError as e:
                print(f"File not found: {e}")
                status, filename = None, None
            except Exception as e:
                print(f"WARNING!!!: An unexpected error occurred while downloading the asset: {e}")
                status, filename = None, None  # Optional fallback values
            #status, filename = download_asset(url)
            #status, filename = download_asset(url.decode())
            if status==200:
                print("    CALIB: %4d/%d dp_id: %s (%s) downloaded"  % (i_calib, len(calib_urls), filename, category))
                #print("    CALIB: %4d/%d dp_id: %s (%s) downloaded"  % (i_calib, len(calib_urls), filename, category.decode()))
            else:
                print("    CALIB: %4d/%d dp_id: %s (%s) NOT DOWNLOADED (http status:%d)"  % (i_calib, len(calib_urls), filename, category, status))
                #print("    CALIB: %4d/%d dp_id: %s (%s) NOT DOWNLOADED (http status:%d)"  % (i_calib, len(calib_urls), filename, category.decode(), status))

        #  ... 5.- PRINT ANY ALERT OR WARNING ENCOUNTERED BY THE PROCESS THAT GENERATES THE CALIBRATION CASCADE
        # ---------------------------------------
        if alert!="":
            print("    %s" % (alert))
        if mode_warning!="":
            print("    %s" % (mode_warning))
        if certified_warning!="":
            print("    %s" % (certified_warning))

        print("------------------------------------------------------------------------------------------------")
    else:
        print("Skipping this download (of\n"+str(calib_urls)+" )\nbecuase skip_csurl and skip_curl are:", skip_csurl, skip_curl)
    print("\n\t### ### ### Completed all the calibration file downloads of this i-th SCIENCE FRAME ### ### ###\n\n")
    ###
print("\t### eso download complete; the cntr_skipCalSelect and cntr_skipCal ended at values of:", cntr_skipCalSelect, cntr_skipCal)
print("\t### The number of calibration files downloaded (len(calib_urls_aggregate)) is:", len(calib_urls_aggregate))
print("\n\t### ### ### COMPLETED all SCIENCE FRAME and their calibration file DOWNLOADS ### ### ###\n\n")

###################################################################################################
# --- 4 Nov 2025: Automatically find matching OBJECT,CENTER frames for this observation/time-window
###################################################################################################
print("Use the code below if there are 0 OBJECT,CENTER calibration frames that have been downloaded...")
there_are_no_object_center_calibrations = 0
if there_are_no_object_center_calibrations == 1:
    print("Searching for matching OBJECT,CENTER calibration frames...\n")

    # --- First, try exact date range
    center_query = f"""
    SELECT dp_id, tpl_name, dp_type, exp_start
    FROM dbo.raw
    WHERE instrument = 'SPHERE'
      AND dp_type LIKE '%CENTER%'
      AND exp_start BETWEEN '{date_start}' AND '{date_end}'
      AND INTERSECTS(s_region, CIRCLE('', {ra}, {dec}, 0.16666667)) = 1
    ORDER BY exp_start
    """

    center_frames = tapobs.search(query=center_query)
    center_table = center_frames.to_table()
    print("There seem to be", len(center_table), "of OBJ/CNTR calib frames\n") 

    # --- If none found, expand search ±1 day
    if len(center_table) == 0:
        print("\n\n⚠️ No OBJECT,CENTER frames found in exact date range — expanding search ±1 day...")
        ds_dt = datetime.strptime(date_start, "%Y-%m-%d %H:%M:%S") - timedelta(days=1)
        de_dt = datetime.strptime(date_end, "%Y-%m-%d %H:%M:%S") + timedelta(days=1)
        new_start = ds_dt.strftime("%Y-%m-%d %H:%M:%S")
        new_end = de_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        print("Searching for matching OBJECT,CENTER calibration frames...")

        center_query = f"""
        SELECT dp_id, tpl_name, dp_type, exp_start
        FROM dbo.raw
        WHERE instrument = 'SPHERE'
          AND dp_type LIKE '%CENTER%'
          AND exp_start BETWEEN '{date_start}' AND '{date_end}'
        ORDER BY exp_start
        """

        center_frames = tapobs.search(query=center_query)
        center_table = center_frames.to_table()
        extra_center_ids = []

        # --- If none found, expand search ±1 day ---
        if len(center_table) == 0:
            print("⚠️ No OBJECT,CENTER frames found in exact date range — expanding search ±1 day...")
            ds_dt = datetime.strptime(date_start, "%Y-%m-%d %H:%M:%S") - timedelta(days=1)
            de_dt = datetime.strptime(date_end, "%Y-%m-%d %H:%M:%S") + timedelta(days=1)
            new_start = ds_dt.strftime("%Y-%m-%d %H:%M:%S")
            new_end = de_dt.strftime("%Y-%m-%d %H:%M:%S")

            center_query_wide = f"""
            SELECT dp_id, tpl_name, dp_type, exp_start
            FROM dbo.raw
            WHERE instrument = 'SPHERE'
              AND dp_type LIKE '%CENTER%'
              AND exp_start BETWEEN '{new_start}' AND '{new_end}'
            ORDER BY exp_start
            """
            center_frames = tapobs.search(query=center_query_wide)
            center_table = center_frames.to_table()
            extra_center_ids = []

    if len(center_table) > 0:
        print(f"✅ Found {len(center_table)} OBJECT,CENTER frame(s):")
        print(center_table)

        # Convert to list for easier handling
        center_rows = list(center_table)

        
        # LIMIT NUMBER OF OBJECT,CENTER FRAMES
        centerFrames_max = 3

        if len(center_rows) > centerFrames_max:
            print(f"\n⚠️ More than {centerFrames_max} OBJECT/CENTER frames found "
                  f"({len(center_rows)}). Selecting the closest {centerFrames_max} in time.")

            from datetime import datetime

            science_times = rawframes['exp_start'].astype(str).tolist()
            #science_times = [str(row['exp_start']) for row in rawframes.to_table()]
            sci_dt = [parse_time(t) for t in science_times]
            median_science_time = sorted(sci_dt)[len(sci_dt)//2]
            
            
            ## science_times already exists earlier in your script - no it doesn't... what was it supposed to be called?
            #sci_dt = [parse_time(t) for t in science_times]
            #median_science_time = sci_dt[len(sci_dt)//2]

            # Sort center frames by closeness to median science exposure
            center_rows = sorted(
                center_rows,
                key=lambda row: abs(parse_time(row['exp_start']) - median_science_time)
            )

            # Keep only 3
            center_rows = center_rows[:centerFrames_max]
            print("Selected CENTER frames:", [r['dp_id'] for r in center_rows],"\n\n")
        else:
            print("Using all CENTER frames.")

        # Save final IDs
        extra_center_ids = [str(row['dp_id']) for row in center_rows]

    else:
        print("⚠️ Still no OBJECT,CENTER frames found, even with ±1 day search.")
        extra_center_ids = []
    print("extra_center_ids variable:", extra_center_ids)

    # Add any extra CENTER frames automatically detected to a new URL to download
    object_center_URL = []
    for dp_id in extra_center_ids:
        object_center_URL += "&dp_id=" + dp_id
    print("\nAdd the 'for dp_id in extra_center_ids' to TheURL =", object_center_URL)


    print("\tFinal TheURL:", TheURL)
    # --- Add automatically detected OBJECT,CENTER frames ---
    if extra_center_ids:
        print("✅ Added OBJECT,CENTER frames automatically:", extra_center_ids)
    else:
        print("ℹ️ No OBJECT,CENTER frames to add.")

    print('\nThe added OBJECT,CENTER frames is:')
    print(object_center_URL)
    
    # --- Download the OBJECT,CENTER frames ---
else:
    print("### Skipped the new code that manually downloads the (missing) OBJECT,CENTER calibration frames")


#############################################################################################################################
# End the script by printing the target name. Capture this printing in the shell script, and feed it into the plotting script
#############################################################################################################################
print(target)
