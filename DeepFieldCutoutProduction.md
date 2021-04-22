# Deep Field Cutout Production

### Get the deep field names

```
_________
DESSCI ~> select distinct field from y6a1_exposure where field like 'SN%';

    |   ☆          |  Ctrl-C to abort;  Rows : 10, Rows/sec: 22 

10 rows in 0.44 seconds

    FIELD
1   SN-X3
2   SN-E2
3   SN-S2
4   SN-X1
5   SN-C2
6   SN-C1
7   SN-E1
8   SN-C3
9   SN-S1
10  SN-X2
```

### Query for the image file paths on NCSA

Run query for each field. Here is X1 for example:

```
select
    SUBSTR(e.filter, 1, 1) as FLT,
	e.mjd_obs,
	i.skybrite,
	e.exptime,
	i.fwhm,
	i.nite,
	fai.path,
	fai.filename,
	fai.compression,
	case 
		when e.mjd_obs < 56400 then 0 
		when e.mjd_obs between 56400.1 and 56800 then 1 
		when e.mjd_obs between 56800.1 and 57100 then 2 
		when e.mjd_obs between 57100.1 and 57500 then 3 
		when e.mjd_obs between 57500.1 and 57900 then 4 
		when e.mjd_obs > 57900.1 then 5 
	end as SEASON
from
	y6a1_proctag t,
	y6a1_image i,
	y6a1_exposure e,
	y6a1_file_archive_info fai
where
	t.tag = 'Y6A1_FINALCUT'
	and t.pfw_attempt_id = i.pfw_attempt_id
	and i.filetype = 'red_immask'
	and i.expnum = e.expnum
	and e.program = 'supernova'
	and e.field = 'SN-X1'
	and i.filename = fai.filename; > X1_metadata.csv
```

Or as one line for easyaccess:
```
select SUBSTR(e.filter, 1, 1) as FLT, e.mjd_obs, i.skybrite, e.exptime, i.fwhm, i.nite, fai.path, fai.filename, fai.compression, case when e.mjd_obs < 56400 then 0 when e.mjd_obs between 56400.1 and 56800 then 1 when e.mjd_obs between 56800.1 and 57100 then 2 when e.mjd_obs between 57100.1 and 57500 then 3 when e.mjd_obs between 57500.1 and 57900 then 4 when e.mjd_obs > 57900.1 then 5 end as SEASON from y6a1_proctag t, y6a1_image i, y6a1_exposure e, y6a1_file_archive_info fai where t.tag='Y6A1_FINALCUT' and t.pfw_attempt_id=i.pfw_attempt_id and i.filetype='red_immask' and i.expnum=e.expnum and e.program='supernova' and e.field='SN-X1' and i.filename=fai.filename; > X1_metadata.csv
```

### Correct SEASON values

Easyaccess seems to have a bug where string values set in a `case` statement can be displayed interactively but do not get written to a file [Issue 181](https://github.com/mgckind/easyaccess/issues/181). 
To account for this bug, the queries save the SEASON as integers and we edit the results with python

After querying, run `fix_metadata_seasons.py`

### Make wget lists

Run the script `make_wget_lists.py`

### Download files

Only download one field and season at a time to conserve disk space. 
The download script will not run if it detects existing images.
The easiest way to clear out existing images is to issue the command `rm -rf images`, since the `images/` directory will be remade automatically.

Below is an example for Y1 X3 images. 
The credentials are your easyaccess credentials and you will be prompted for your password.

```
python download_images.py --field X3 --season Y1 --username <username>
```

To check progress while the download is running, you can use the same command, but append `--check_progress` to it.

```
python download_images.py --field X3 --season Y1 --username <username> --check_progress
```

### Make the catalog

`mkdir catalog && cd catalog`

Run the query:

```
select
	o.ra,
	o.dec,
	o.tilename,
	s.mag_auto_g,
	s.mag_auto_r,
	s.mag_auto_i,
	s.mag_auto_z,
	s.coadd_object_id,
	o.psf_mag_i,
	o.cm_mag_i,
	o.cm_t,
	s.WAVG_SPREAD_MODEL_I,
	s.SPREAD_MODEL_I,
	s.CLASS_STAR_I
from
	Y3A2_MISC_MOF_V1_COADD_TRUTH o,
	Y3A2_MISC_COADD_OBJECT_SUMMARY s
where
	o.coadd_object_id = s.coadd_object_id
	and s.mag_auto_g < 27.5
	and s.mag_auto_r < 27.5
	and s.mag_auto_i < 27.5
	and s.mag_auto_z < 27.5; > deep_cal.csv
```

Again, one line version for easyaccess:
```
select o.ra, o.dec, o.tilename, s.mag_auto_g, s.mag_auto_r, s.mag_auto_i, s.mag_auto_z, s.coadd_object_id, o.psf_mag_i, o.cm_mag_i, o.cm_t, s.WAVG_SPREAD_MODEL_I, s.SPREAD_MODEL_I, s.CLASS_STAR_I from Y3A2_MISC_MOF_V1_COADD_TRUTH o, Y3A2_MISC_COADD_OBJECT_SUMMARY s where o.coadd_object_id = s.coadd_object_id and s.mag_auto_g < 27.5 and s.mag_auto_r < 27.5 and s.mag_auto_i < 27.5 and s.mag_auto_z < 27.5; > deep_cal.csv
```

Then run `make_catalog.py` to apply a star-galaxy separation cut and merge the resulting files into one catalog named `deep_catalog.csv`.

### Produce cutouts

`cd Y6_Bulk_Coadd_Cutouts`

The script `cutout.py` has the following arguments:

```
$ python cutout.py --help
usage: cutout.py [-h] --field FIELD --ccd CCD --season SEASON
                 [--outdir OUTDIR] [--size SIZE] [--test] [--maglim MAGLIM]

Single-epoch deep-field cutout production

optional arguments:
  -h, --help       show this help message and exit
  --field FIELD    Like X1, X3, C3, etc.
  --ccd CCD        CCD number as integer
  --season SEASON  Like SV, Y1, Y2, etc.
  --outdir OUTDIR  Directory for results
  --size SIZE      Side length (in px) of cutouts
  --test           Run on only 25 objects and 2 nites
  --maglim MAGLIM  Faintest i-magnitude to include
```

So to run on Y1 X3 images for CCD 33, the command would be:

`python cutout.py --season Y1 --field X3 --ccd 33 --outdir /data/des81.b/data/stronglens/DEEP_FIELDS/CUTOUTS/Y1/X3/ --maglim 23.5`

### Produce cutouts for all CCDs in a chosen field and season

`cd job_sub/`

First check that all nodes are ready by running the command below and looking in the `status/` directory.

`python run_production.py --field X3 --season Y1 --test`

Then, run the production, specifying the desired maglim (default is 90)

`python run_production.py --field X3 --season Y1 --maglim 23.5`

While the jobs are running, track progress with 

`python progress_tracker.py`