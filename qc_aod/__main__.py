#*******************************************************************************
#  Author:
#     name:  Erol Cromwell
#     phone: 509-372-4648
#     email: erol.cromwell@pnnl.gov
#*******************************************************************************

import sys
import os
from time import gmtime, strftime
from .version import __version__
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

try:
    import dsproc3 as dsproc
    import cds3 as cds
except ImportError as e:
    print("Could not import ADI")

from . import qc_aod_input_fields as input
from . import qc_aod_trans_fields as trans
from . import qc_aod_output_fields as output

MISSING_VALUE = -9999

class UserData(object):
    def __init__(self, proc_name):
        self.proc_name = proc_name
        self.out_dsid = ''


#  Get variable data
#
#  @param dataset   - CDSGroup dataset
#  @param var_name  - Name of variable
#  @param data_type - Variable data type 
#  @return
#    - Variable data
#    - NOne if error occured
def get_var_data(dataset, var_name, data_type):

    var = dsproc.get_var(dataset, var_name)
    if var is None:
        error_string = "Required variable {} not found in dataset".format(var_name)
        dsproc.error(dsproc.EREQVAR, error_string) 
        return None
        
    var_data, missing = dsproc.get_var_data(var, data_type, 0)
    if var_data is None:
        return None

    return var_data



# Initialize the process.
#  This function is used to do any up front process initialization that
#  is specific to this process, and to create the UserData structure that
#  will be passed to all hook functions.
#  If an error occurs in this function it will be appended to the log and
#  error mail messages, and the process status will be set appropriately.
#  @return
#    - a user defined data structure or value (a Python object) that 
#      will be passed in as user_data to all other hook functions.
#    - 1 if no user data is returned.
#    - None if fatal error occurred and the process should exit.

def init_process():
    dsproc.debug_lv1("*********Inside Init Hook***********\n")

    dsproc.debug_lv1("Creating user defined data structure\n")
    mydata = UserData(dsproc.get_name())

    mydata.out_dsid = dsproc.get_output_datastream_id("qcaod", "c1")
    
    if mydata.out_dsid < 0:
        return None

    # Set the process interval to be solar day 
    site_tz = -6.0

    dsproc.debug_lv2('Adjusting procesing interval for SGP timezone by %f hours'.format(site_tz) )

    tz_offset = site_tz * -3600.0
    dsproc.set_processing_interval_offset(tz_offset)


    # Load list of bad nimfr days, if exists

    conf_home = os.getenv("VAP_HOME")
    fname = os.path.join(conf_home, "conf", "vap", "qc_aod", "bad_nimfr_days.txt")
    
    bad_nimfr_days = []

    if os.path.exists(fname):
        # Day are in format YYYYMMDD
        # Or, date range of YYYYMMDD-YYYYMMDD, is inclusive
        tmp_bad_nimfr_days = []
        with open(fname, 'r') as f:
            for line in f:
                
                # Skip lines with comment (i.e #)
                if line[0] == '#':
                    continue
                
                day = line.strip('\n')
                tmp_bad_nimfr_days.append(day)

        # For each entry, check if day or date range
        for entry in tmp_bad_nimfr_days:
            
            if len(entry) == 8:
                bad_nimfr_days.append(entry)
            else:
                start_date = entry[:8]
                end_date = entry[9:]

                start = datetime.datetime.strptime(start_date, '%Y%m%d')
                end = datetime.datetime.strptime(end_date, '%Y%m%d')

                date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days+1)]
                for date in date_generated:
                    bad_nimfr_days.append(date.strftime('%Y%m%d'))

    mydata.bad_nimfr_days = bad_nimfr_days

    return mydata



#  Finish the process.
#  This function frees all memory used by the UserData structure.
#  @param user_data  - void pointer to the UserData structure
def finish_process(user_data):
    dsproc.debug_lv1("*********Inside Finish Hook***********\n")

    dsproc.debug_lv1("Cleaning up user defined data structure for process\n")



#  Hook function called just prior to data transformation.
#  This function will be called once per processing interval just prior to
#  data transformation, and after the retrieved observations are merged.

#  @param user_data  - void pointer to the UserData structure
#                      returned by the init_process() function
#  @param begin_date - the begin time of the current processing interval
#  @param end_date   - the end time of the current processing interval
#  @param ret_data   - pointer to the parent CDSGroup containing all the
#  @return
#    -  1 if processing should continue normally
#    -  0 if processing should skip the current processing interval
#         and continue on to the next one.
#    - -1 if a fatal error occurred and the process should exit.
def pre_transform_hook(user_data, begin_date, end_date, ret_data):

    status = 1

    begin_time = strftime('%Y%m%d.%H%M%S', gmtime(begin_date))
    end_time = strftime('%Y%m%d.%H%M%S', gmtime(end_date))

    FAIL_MIN = 0
    WARN_MAX = 1

    # Get QC, set every intermediata/bad value to bad
    CDS_INT   = 4
    CDS_FLOAT = 5

    ret_groups = ret_data.get_groups()
    group_names = [ g.get_name() for g in ret_groups ]

    wavelengths = [500, 870]
    
    # For each one, set values above threshold (10) to missing value
    for top_group in ret_groups:

        name = top_group.get_name()
        dsproc.debug_lv1("Retrieving: {}".format( name ) )
        
        if 'csphot' in name or 'aeronet' in name:
            var_name_prefix = 'aod_cimel'
        elif 'sgpmfrsraod1michC1' in name:
            var_name_prefix = 'aod_mfrsr_C1'
        elif 'sgpmfrsraod1michE13' in name:
            var_name_prefix = 'aod_mfrsr_E13'
        else:
            var_name_prefix = 'aod_nimfr'

        sub_groups = top_group.get_groups()
        sub_group_with_data = sub_groups[0]

        for wavelength in wavelengths:
            var_name = "{}_{}".format(var_name_prefix, wavelength)
        
            var = dsproc.get_var(sub_group_with_data, var_name)
            if var is None:
                error_string = "Required variable {} not found in dataset".format(var_name)
                dsproc.error(dsproc.EREQVAR, error_string) 
                return -1
                
            var_data, missing = dsproc.get_var_data(var, CDS_FLOAT, 0)


            # Below 0, Above 1
            below_fail_min = var_data < FAIL_MIN
            above_warn_max = var_data > WARN_MAX

            var_data[below_fail_min] = MISSING_VALUE
            var_data[above_warn_max] = MISSING_VALUE
                
            nsamples = len(var_data)


            status = dsproc.set_var_data(var, 0, nsamples, None, var_data)
            if status is None:
                return -1

            qc_var = dsproc.get_qc_var(var)
            if qc_var is not None:
                qc_var_data, missing = dsproc.get_var_data(qc_var, CDS_INT, 0) 

                # Testing setting anything not good, as bad
                not_good_qc = qc_var_data != 0
                qc_var_data[not_good_qc] = 1

                # Below 0, Above 1
                qc_var_data[below_fail_min] = 1
                qc_var_data[above_warn_max] = 1

                status = dsproc.set_var_data(qc_var, 0, nsamples, None, qc_var_data)
                if status is None:
                    return -1




    # If only one value for cimel, artificially add a sample to make 
    # transformation work
    # Note, this is a hack and not currently supported by ADI
    cimel_name = ""

    for name in group_names:
        if 'csphot' in name:
            cimel_name = name
            break

    var_name_prefix = 'aod_cimel'
    if cimel_name:

        dsproc.debug_lv1("Retrieving: {}".format(cimel_name) )
        top_group = ret_data.get_group(cimel_name)
        
        if not top_group:
            dsproc.debug_lv1("No retrieved group for {}, skipping".format(cimel_name))
        else:
        
            sub_groups = top_group.get_groups()
            sub_group_with_data = sub_groups[0]

            # 500nm cimel data
            var_name_500 = "{}_500".format(var_name_prefix, wavelength)
            
            var_500 = dsproc.get_var(sub_group_with_data, var_name_500)
            if var_500 is None:
                error_string = "Required variable {} not found in dataset".format(var_name_500)
                dsproc.error(dsproc.EREQVAR, error_string) 
                return -1
                
            var_500_data, missing = dsproc.get_var_data(var_500, CDS_FLOAT, 0)

            # Check if have only one value
            nsamples_500 = len(var_500_data)

            # 870nm cimel var
            var_name_870 = "{}_870".format(var_name_prefix, wavelength)
            
            var_870 = dsproc.get_var(sub_group_with_data, var_name_870)
            if var_870 is None:
                error_string = "Required variable {} not found in dataset".format(var_name_870)
                dsproc.error(dsproc.EREQVAR, error_string) 
                return -1
                
            if nsamples_500 == 1:
                
                dsproc.debug_lv1("Only 1 cimel sample, artificially adding another")
                dsproc.log("Only 1 cimel sample, artificially adding another")

                times = dsproc.get_sample_times(sub_group_with_data, 0)
                times[0] = times[0] + 1

                dsproc.set_sample_times(sub_group_with_data, 1, times)

                var_type = var_500_data.dtype
                fake_value = [MISSING_VALUE]
                fake_value = np.array(fake_value)
                fake_value = fake_value.astype(var_type)

                status = dsproc.set_var_data(var_500, 1, 1, None, fake_value)
                if status is None:
                    return -1

                status = dsproc.set_var_data(var_870, 1, 1, None, fake_value)
                if status is None:
                    return -1


    
    if dsproc.get_debug_level() > 1:
        dsproc.dump_retrieved_datasets("./debug_dumps", "retrieved_data.debug", 0)

    return 1



#  Hook function called just after data transformation.
#  This function will be called once per processing interval just after data
#  transformation, but before the process_data function is called.
#  @param  user_data - void pointer to the UserData structure
#                      returned by the init_process() function
#  @param begin_date - the begin time of the current processing interval
#  @param end_date   - the end time of the current processing interval
#  @param trans_data - pointer to the parent CDSGroup containing all the
#  @return
#    -  1 if processing should continue normally
#    -  0 if processing should skip the current processing interval
#         and continue on to the next one.
#    - -1 if a fatal error occurred and the process should exit.

def post_transform_hook(user_data, begin_date, end_date, input_data):
    status = 1

#    Example function call
#    status = <process>_post_transform_hook(
#            user_data, begin_date, end_date, input_data)


    return status

#  Get transformed variabled data
#
#  @param var_name  - Name of transformed variabel
#  @param data_type - Variable data type 
#  @return
#    - Transformed variable data
#    - [] if no data found
#    - None if error occured
def get_transformed_var_data(var_name, data_type):

    var = dsproc.get_transformed_var(var_name, 0)
    if var is None:
        warning_string = "Required variable {} not found in transformed dataset".format(var_name)
        dsproc.warning(warning_string) 
        return []
        
    var_data, missing = dsproc.get_var_data(var, data_type, 0)
    if var_data is None:
        return None

    return var_data

#  Initalize variable data
#
#  @param dataset   - CDSGroup dataset
#  @param var_name  - Name of transformed variabel
#  @param start     - Start of sample
#  @param nsamples  - Number of samples
#  @return
#    - Variable data
#    - NOne if error occured
def init_var_data(dataset, var_name, start, nsamples, use_missing=False):

    var = dsproc.get_var(dataset, var_name)
    if var is None:
        error_string = "Required variable {} not found in dataset".format(var_name)
        dsproc.error(dsproc.EREQVAR, error_string) 
        return None
        
    var_data = dsproc.init_var_data(var, start, nsamples, use_missing)
    if var_data is None:
        return None

    return var_data

#  Ser variable attribute value
#
#  @param dataset    - CDSGroup dataset
#  @param var_name   - Name of variable
#  @param att_name   - Name of attribute
#  @param cds_type   - CDS Data Type
#  @param att_value  - Attribute value
#  @return
#    - 1 if successful
#    - 0 if an error occured
def set_var_att_value(dataset, var_name, att_name, cds_type, att_value):

    var = dsproc.get_var(dataset, var_name)
    if var is None:
        error_string = "Required variable {} not found in dataset".format(var_name)
        dsproc.error(dsproc.EREQVAR, error_string) 
        return 0

    status = dsproc.set_att_value(var, att_name, cds_type, att_value) 

    return status



#  Calculate following statistics from set of aod data:
#       - Daily mean
#       - standard deviation
#       - number of points of good points
#
#
#  @param aod_data -  AOD data
#  @param qc_aod_data -  QC AOD data
#
#  @return
#    - mean, std, npoints
#    - None if an error occurs
def calc_aod_mean(aod_data, good_aod_index):

    good_aod_data = aod_data[good_aod_index]
    aod_points = int(len(good_aod_data))

    aod_mean = MISSING_VALUE
    aod_std = MISSING_VALUE
    if aod_points > 0:
        aod_mean = np.mean(good_aod_data)

        if aod_points > 1:
            # ddof means divided by N - ddof
            aod_std = np.std(good_aod_data, ddof=1)

    return aod_mean, aod_std, aod_points 




#  Calculate following statistics for two sets of aod data:
#       - mean bias
#       - RMSD
#       - slope
#       - number of points for comparison
#
#  Data has already been cleaned, with qc
#
#  @param aod_data_1 - First aod data
#  @param aod_data_2 - Second aod data
#  @param has_cimel  - Whether one of the data is cimel
#  @return
#    - mean bias, RMSD, slope, and number of points for comparison, whether passes checks
#    - None if an error occurs
def compare_aod_statistics(aod_data_1, aod_data_2, has_cimel=False):

    if has_cimel:
        npoints_threshold = 10
    else:
        npoints_threshold = 100

    mean_bias_threshold = 0.02
    r2_threshold = 0.9
    slope_threshold = 0.2

    # Number of points for comparison
    npoints = len(aod_data_1)

    # If filtered data empty, return missing values
    if npoints == 0:
        return MISSING_VALUE, MISSING_VALUE, MISSING_VALUE, npoints, False

    if len(aod_data_1) > 1:
        y_train = aod_data_1.reshape(-1,1)
    else:
        y_train = aod_data_1.reshape(1, -1)

    if len(aod_data_2) > 1:
        x_train = aod_data_2.reshape(-1,1)
    else:
        x_train = aod_data_2.reshape(1, -1)

    xy_regress = LinearRegression() 
    xy_regress.fit(x_train, y_train)
    r2 = xy_regress.score(x_train, y_train)
    slope = xy_regress.coef_[0][0]

    diff = aod_data_1 - aod_data_2
    mean_bias = round( np.mean(diff), 3)

    npoint_check = npoints >= npoints_threshold
    bias_check   = abs(mean_bias) <= mean_bias_threshold
    r2_check     = abs(r2) >= r2_threshold
    slope_check  = abs(slope - 1) <= slope_threshold 

    good_compare = npoint_check and bias_check and r2_check and slope_check

    return mean_bias, r2, slope, npoints, good_compare

#  Main data processing function.
#  This function will be called once per processing interval just after the
#  output datasets are created, but before they are stored to disk.

#  @param  user_data - void pointer to the UserData structure
#                      returned by the init_process() function
#  @param  begin_date - begin time of the processing interval
#  @param  end_date   - end time of the processing interval
#  @param  trans_data - retriever data transformed to user defined 
#  @return
#    -  1 if processing should continue normally
#    -  0 if processing should skip the current processing interval
#         and continue on to the next one.
#    - -1 if a fatal error occurred and the process should exit.

def process_data(proc_data, begin_date, end_date, input_data):
    data = proc_data

    begin_time = strftime('%Y%m%d.%H%M%S', gmtime(begin_date))
    end_time = strftime('%Y%m%d.%H%M%S', gmtime(end_date))
    dsproc.debug_lv1("begin_date = " + begin_time)
    dsproc.debug_lv1("end_date = " + end_time)

    bad_nimfr_days = data.bad_nimfr_days
    begin_day = strftime('%Y%m%d', gmtime(begin_date))
    # -------------------------------------------------------------
    # Start algorithm
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # get output dataset
    # -------------------------------------------------------------
    out_ds = dsproc.get_output_dataset(data.out_dsid, 0)

    # Run the qc_limit_checks on the variables
    var_names = [ "aod_mfrsr_C1_500",
                  "aod_mfrsr_E13_500",
                  "aod_nimfr_500",
                  "aod_cimel_500",
                  "aod_mfrsr_C1_870",
                  "aod_mfrsr_E13_870",
                  "aod_nimfr_870",
                  "aod_cimel_870" ]

    # -------------------------------------------------------------
    # Get transformed variables and data
    # QC data is not required, but wanted, don't fail if don't find
    # any
    # -------------------------------------------------------------
    CDS_INT = 4
    CDS_FLOAT = 5

    # Shape to None
    shape = None

    # -------------------------------------------------------------
    # 500 nm data
    # -------------------------------------------------------------

    # aod_cimel_500
    aod_cimel_500 = get_transformed_var_data("aod_cimel_500", CDS_FLOAT)
    if aod_cimel_500 is None:
        return(-1)
    elif aod_cimel_500 != []:
        shape = aod_cimel_500.shape

    # aod_nimfr_500
    aod_nimfr_500 = get_transformed_var_data("aod_nimfr_500", CDS_FLOAT)
    if aod_nimfr_500 is None:
        return(-1)
    elif aod_nimfr_500 != []:
        shape = aod_nimfr_500.shape

    # qc_aod_nimfr_500
    qc_aod_nimfr_500 = get_var_data(out_ds, "qc_aod_nimfr_500", CDS_FLOAT)

    # aod_mfrsr_E13_500
    aod_mfrsr_E13_500 = get_transformed_var_data("aod_mfrsr_E13_500", CDS_FLOAT)
    if aod_mfrsr_E13_500 is None:
        return(-1)
    elif aod_mfrsr_E13_500 != []:
        shape = aod_mfrsr_E13_500.shape

    # qc_aod_mfrsr_E13_500
    qc_aod_mfrsr_E13_500 = get_var_data(out_ds, "qc_aod_mfrsr_E13_500", CDS_FLOAT)

    # aod_mfrsr_E13 good fraction
    #aod_mfrsr_E13_500_goodfraction = get_var_data(out_ds, "aod_mfrsr_E13_500_goodfraction", CDS_FLOAT)

    # aod_mfrsr_C1_500
    aod_mfrsr_C1_500 = get_transformed_var_data("aod_mfrsr_C1_500", CDS_FLOAT)
    if aod_mfrsr_C1_500 is None:
        return(-1)
    elif aod_mfrsr_C1_500 != []:
        shape = aod_mfrsr_C1_500.shape

    # qc_aod_mfrsr_C1_500
    qc_aod_mfrsr_C1_500 = get_var_data(out_ds, "qc_aod_mfrsr_C1_500", CDS_FLOAT)

    # aod_mfrsr_C1 good fraction
    # aod_mfrsr_C1_500_goodfraction = get_var_data(out_ds, "aod_mfrsr_C1_500_goodfraction", CDS_FLOAT)

    # -------------------------------------------------------------
    # 870 nm data
    # -------------------------------------------------------------

    # aod_cimel_870
    aod_cimel_870 = get_transformed_var_data("aod_cimel_870", CDS_FLOAT)
    if aod_cimel_870 is None:
        return(-1)
    elif aod_cimel_870 != []:
        shape = aod_cimel_870.shape

    # aod_nimfr_870
    aod_nimfr_870 = get_transformed_var_data("aod_nimfr_870", CDS_FLOAT)
    if aod_nimfr_870 is None:
        return(-1)
    elif aod_nimfr_870 != []:
        shape = aod_nimfr_870.shape

    # qc_aod_nimfr_870
    qc_aod_nimfr_870 = get_var_data(out_ds, "qc_aod_nimfr_870", CDS_FLOAT)

    # aod_mfrsr_E13_870
    aod_mfrsr_E13_870 = get_transformed_var_data("aod_mfrsr_E13_870", CDS_FLOAT)
    if aod_mfrsr_E13_870 is None:
        return(-1)
    elif aod_mfrsr_E13_870 != []:
        shape = aod_mfrsr_E13_870.shape

    # qc_aod_mfrsr_E13_870
    qc_aod_mfrsr_E13_870 = get_var_data(out_ds, "qc_aod_mfrsr_E13_870", CDS_FLOAT)

    # aod_mfrsr_C1_870
    aod_mfrsr_C1_870 = get_transformed_var_data("aod_mfrsr_C1_870", CDS_FLOAT)
    if aod_mfrsr_C1_870 is None:
        return(-1)
    elif aod_mfrsr_C1_870 != []:
        shape = aod_mfrsr_C1_870.shape

    # qc_aod_mfrsr_C1_870
    qc_aod_mfrsr_C1_870 = get_var_data(out_ds, "qc_aod_mfrsr_C1_870", CDS_FLOAT)

    # Shape never set, skip day since no data available 
    if shape is None:
        dsproc.log("No input AOD data available, skipping day")
        return(0)

 

    # For missing data, set to missing value
    if aod_cimel_500 == []:
        aod_cimel_500 = np.ones(shape, dtype=np.float32) * MISSING_VALUE               

    if aod_nimfr_500 == []:
        aod_nimfr_500 = np.ones(shape, dtype=np.float32) * MISSING_VALUE               

    if aod_mfrsr_E13_500 == []:
        aod_mfrsr_E13_500 = np.ones(shape, dtype=np.float32) * MISSING_VALUE               

    if aod_mfrsr_C1_500 == []:
        aod_mfrsr_C1_500 = np.ones(shape, dtype=np.float32) * MISSING_VALUE               

    if aod_cimel_870 == []:
        aod_cimel_870 = np.ones(shape, dtype=np.float32) * MISSING_VALUE               

    if aod_nimfr_870 == []:
        aod_nimfr_870 = np.ones(shape, dtype=np.float32) * MISSING_VALUE               

    if aod_mfrsr_E13_870 == []:
        aod_mfrsr_E13_870 = np.ones(shape, dtype=np.float32) * MISSING_VALUE               

    if aod_mfrsr_C1_870 == []:
        aod_mfrsr_C1_870 = np.ones(shape, dtype=np.float32) * MISSING_VALUE               


    # -------------------------------------------------------------
    # Initalize variable data
    # -------------------------------------------------------------

    # Daily statistic variables
    daily_RMSD_500 = init_var_data(out_ds, "daily_RMSD_500", 0, 6, use_missing=True)
    if daily_RMSD_500 is None:
        return(-1)

    daily_npoint_500 = init_var_data(out_ds, "daily_npoint_500", 0, 6, use_missing=True)
    if daily_npoint_500 is None:
        return(-1)

    daily_mean_bias_500 = init_var_data(out_ds, "daily_mean_bias_500", 0, 6, use_missing=True)
    if daily_mean_bias_500 is None:
        return(-1)

    daily_slope_500 = init_var_data(out_ds, "daily_slope_500", 0, 6, use_missing=True)
    if daily_slope_500 is None:
        return(-1)

    # -------------------------------------------------------------
    # Compare statistics
    # -------------------------------------------------------------
    good_data_flag = 0
    use_aod_mfrsr_C1_500 = False
    use_aod_mfrsr_E13_500 = False
    use_aod_nimfr_500 = False
    use_aod_cimel_500 = False


    QC_SOME_BAD_INPUTS = 6
    QC_INDETERMINATE= 2
    QC_SOME_BAD_INPUTS_BITS = np.power(2, QC_SOME_BAD_INPUTS-1)
    QC_INDETERMINATE_BITS = np.power(2, QC_INDETERMINATE-1)


    AT_LEAST_ONE = QC_INDETERMINATE_BITS + QC_SOME_BAD_INPUTS_BITS
    ###################
    # Location of filtered data
    # CIMEL has no qc data
    ###################
    
    # QC for 500 nm
    if qc_aod_mfrsr_C1_500 is not None:
        good_mfrsr_C1_500_index = (qc_aod_mfrsr_C1_500 == 0) * (aod_mfrsr_C1_500 != MISSING_VALUE)
        some_bad_mfrsr_C1_index = (qc_aod_mfrsr_C1_500 == QC_SOME_BAD_INPUTS_BITS)

        good_mfrsr_C1_500_index = ( good_mfrsr_C1_500_index + some_bad_mfrsr_C1_index ) > 0

    else:
        good_mfrsr_C1_500_index = aod_mfrsr_C1_500 != MISSING_VALUE

    if qc_aod_mfrsr_E13_500 is not None:
        good_mfrsr_E13_500_index = (qc_aod_mfrsr_E13_500 == 0) * (aod_mfrsr_E13_500 != MISSING_VALUE)
        some_bad_mfrsr_E13_index = (qc_aod_mfrsr_E13_500 == QC_SOME_BAD_INPUTS_BITS)

        good_mfrsr_E13_500_index = ( good_mfrsr_E13_500_index + some_bad_mfrsr_E13_index ) > 0

    else:
        good_mfrsr_E13_500_index = aod_mfrsr_E13_500 != MISSING_VALUE
        
    if qc_aod_nimfr_500 is not None:
        good_nimfr_500_index = (qc_aod_nimfr_500 == 0) * (aod_nimfr_500 != MISSING_VALUE)
        some_bad_nimfr_index = (qc_aod_nimfr_500 == QC_SOME_BAD_INPUTS_BITS)

        good_nimfr_500_index = ( good_nimfr_500_index + some_bad_nimfr_index ) > 0

    else:
        good_nimfr_500_index = aod_nimfr_500 != MISSING_VALUE
    
    good_cimel_500_index = aod_cimel_500 != MISSING_VALUE


    # QC for 870 nm
    if qc_aod_mfrsr_C1_870 is not None:
        good_mfrsr_C1_870_index = (qc_aod_mfrsr_C1_870 == 0) * (aod_mfrsr_C1_870 != MISSING_VALUE)
        some_bad_mfrsr_C1_index = (qc_aod_mfrsr_C1_870 == QC_SOME_BAD_INPUTS_BITS)

        good_mfrsr_C1_870_index = ( good_mfrsr_C1_870_index + some_bad_mfrsr_C1_index ) > 0

    else:
        good_mfrsr_C1_870_index = aod_mfrsr_C1_870 != MISSING_VALUE

    if qc_aod_mfrsr_E13_870 is not None:
        good_mfrsr_E13_870_index = (qc_aod_mfrsr_E13_870 == 0) * (aod_mfrsr_E13_870 != MISSING_VALUE)
        some_bad_mfrsr_E13_index = (qc_aod_mfrsr_E13_870 == QC_SOME_BAD_INPUTS_BITS)

        good_mfrsr_E13_870_index = ( good_mfrsr_E13_870_index + some_bad_mfrsr_E13_index ) > 0

    else:
        good_mfrsr_E13_870_index = aod_mfrsr_E13_870 != MISSING_VALUE
        
    if qc_aod_nimfr_870 is not None:
        good_nimfr_870_index = (qc_aod_nimfr_870 == 0) * (aod_nimfr_870 != MISSING_VALUE)
        some_bad_nimfr_index = (qc_aod_nimfr_870 == QC_SOME_BAD_INPUTS_BITS)

        good_nimfr_870_index = ( good_nimfr_870_index + some_bad_nimfr_index ) > 0

    else:
        good_nimfr_870_index = aod_nimfr_870 != MISSING_VALUE
    
    good_cimel_870_index = aod_cimel_870 != MISSING_VALUE

    # -------------------------------------------------------------
    # Calculate daily mean, std, and number of points for each AOD measurement
    # -------------------------------------------------------------
                     
    aod_data_list = [ aod_mfrsr_C1_500,
                      aod_mfrsr_E13_500,
                      aod_nimfr_500,
                      aod_cimel_500,
                      aod_mfrsr_C1_870,
                      aod_mfrsr_E13_870,
                      aod_nimfr_870,
                      aod_cimel_870 ]


    good_aod_indices = [ good_mfrsr_C1_500_index,
                         good_mfrsr_E13_500_index,
                         good_nimfr_500_index,
                         good_cimel_500_index,
                         good_mfrsr_C1_870_index,
                         good_mfrsr_E13_870_index,
                         good_nimfr_870_index,
                         good_cimel_870_index ]

    for i in range(len(aod_data_list)):
        aod_data = aod_data_list[i]
        good_aod_index = good_aod_indices[i]

        aod_mean, aod_std, npoints = calc_aod_mean(aod_data, good_aod_index)
        var_name = var_names[i]
        
        status = set_var_att_value(out_ds, var_name, "daily_mean", CDS_FLOAT, aod_mean)
        if status == 0:
            return -1

        status = set_var_att_value(out_ds, var_name, "daily_std", CDS_FLOAT, aod_std)
        if status == 0:
            return -1

        status = set_var_att_value(out_ds, var_name, "number_good_points", CDS_FLOAT, npoints)
        if status == 0:
            return -1


    # -------------------------------------------------------------
    # Compare aod_mfrsr_C1 and aod_mfrsr_E13
    # -------------------------------------------------------------
    dsproc.debug_lv1("Comparing aod_mfrsr_C1 and aod_mfrsr_E13")

    daily_index = 0

    good_data =  good_mfrsr_C1_500_index * good_mfrsr_E13_500_index

    good_aod_mfrsr_C1_500 = aod_mfrsr_C1_500[good_data]
    good_aod_mfrsr_E13_500 = aod_mfrsr_E13_500[good_data]

    mean_bias, r2, slope, npoints, good_compare = compare_aod_statistics(good_aod_mfrsr_C1_500, good_aod_mfrsr_E13_500)

    daily_RMSD_500[daily_index]  = r2
    daily_npoint_500[daily_index]  = npoints
    daily_mean_bias_500[daily_index]  = mean_bias
    daily_slope_500[daily_index] = slope

    if good_compare:
        use_aod_mfrsr_C1_500 = True
        use_aod_mfrsr_E13_500 = True
 
    # -------------------------------------------------------------
    # Compare aod_mfrsr_C1 and aod_nimfr
    # -------------------------------------------------------------
    dsproc.debug_lv1("Comparing aod_mfrsr_C1 and aod_nimfr")
    daily_index = 1


    # -------------------------------------------------------------
    # Compare aod_mfrsr_C1 and aod_mfrsr_E13
    # -------------------------------------------------------------
    dsproc.debug_lv1("Comparing aod_mfrsr_C1 and aod_mfrsr_E13")

    daily_index = 0

    good_data =  good_mfrsr_C1_500_index * good_mfrsr_E13_500_index

    good_aod_mfrsr_C1_500 = aod_mfrsr_C1_500[good_data]
    good_aod_mfrsr_E13_500 = aod_mfrsr_E13_500[good_data]

    mean_bias, r2, slope, npoints, good_compare = compare_aod_statistics(good_aod_mfrsr_C1_500, good_aod_mfrsr_E13_500)

    daily_RMSD_500[daily_index]  = r2
    daily_npoint_500[daily_index]  = npoints
    daily_mean_bias_500[daily_index]  = mean_bias
    daily_slope_500[daily_index] = slope

    if good_compare:
        use_aod_mfrsr_C1_500 = True
        use_aod_mfrsr_E13_500 = True
 
    # -------------------------------------------------------------
    # Compare aod_mfrsr_C1 and aod_nimfr
    # -------------------------------------------------------------
    dsproc.debug_lv1("Comparing aod_mfrsr_C1 and aod_nimfr")
    daily_index = 1

    good_data =  good_mfrsr_C1_500_index * good_nimfr_500_index

    good_aod_mfrsr_C1_500 = aod_mfrsr_C1_500[good_data]
    good_aod_nimfr_500 = aod_nimfr_500[good_data]

    mean_bias, r2, slope, npoints, good_compare = compare_aod_statistics(good_aod_mfrsr_C1_500, good_aod_nimfr_500)

    daily_RMSD_500[daily_index]  = r2
    daily_npoint_500[daily_index]  = npoints
    daily_mean_bias_500[daily_index]  = mean_bias
    daily_slope_500[daily_index] = slope

    if good_compare:
        use_aod_mfrsr_C1_500 = True
        use_aod_nimfr_500 = True
 
    # -------------------------------------------------------------
    # Compare aod_mfrsr_C1 and aod_cimel
    # -------------------------------------------------------------
    dsproc.debug_lv1("Comparing aod_mfrsr_C1 and aod_cimel")
    daily_index = 2

    good_data =  good_mfrsr_C1_500_index * good_cimel_500_index 

    good_aod_mfrsr_C1_500 = aod_mfrsr_C1_500[good_data]
    good_aod_cimel_500 = aod_cimel_500[good_data]

    mean_bias, r2, slope, npoints, good_compare = compare_aod_statistics(good_aod_mfrsr_C1_500, good_aod_cimel_500,
                                                                         has_cimel = True)

    daily_RMSD_500[daily_index]  = r2
    daily_npoint_500[daily_index]  = npoints
    daily_mean_bias_500[daily_index]  = mean_bias
    daily_slope_500[daily_index] = slope

    if good_compare:
        use_aod_mfrsr_C1_500 = True
        use_aod_cimel_500 = True

    # -------------------------------------------------------------
    # Compare aod_mfrsr_E13 and aod_nimfr
    # -------------------------------------------------------------
    dsproc.debug_lv1("Comparing aod_mfrsr_E13 and aod_nimfr")

    daily_index = 3

    good_data =  good_mfrsr_E13_500_index * good_nimfr_500_index

    good_aod_mfrsr_E13_500 = aod_mfrsr_E13_500[good_data]
    good_aod_nimfr_500 = aod_nimfr_500[good_data]

    mean_bias, r2, slope, npoints, good_compare = compare_aod_statistics(good_aod_mfrsr_E13_500, good_aod_nimfr_500)

    daily_RMSD_500[daily_index]  = r2
    daily_npoint_500[daily_index]  = npoints
    daily_mean_bias_500[daily_index]  = mean_bias
    daily_slope_500[daily_index] = slope

    if good_compare:
        use_aod_mfrsr_E13_500 = True
        use_aod_nimfr_500 = True
 
    # -------------------------------------------------------------
    # Compare aod_mfrsr_E13 and aod_cimel
    # -------------------------------------------------------------
    dsproc.debug_lv1("Comparing aod_mfrsr_E13 and aod_cimel")

    daily_index = 4

    good_data =  good_mfrsr_E13_500_index * good_cimel_500_index 

    good_aod_mfrsr_E13_500 = aod_mfrsr_E13_500[good_data]
    good_aod_cimel_500 = aod_cimel_500[good_data]

    mean_bias, r2, slope, npoints, good_compare = compare_aod_statistics(good_aod_mfrsr_E13_500, good_aod_cimel_500,
                                                                         has_cimel = True)

    daily_RMSD_500[daily_index]  = r2
    daily_npoint_500[daily_index]  = npoints
    daily_mean_bias_500[daily_index]  = mean_bias
    daily_slope_500[daily_index] = slope

    if good_compare:
        use_aod_mfrsr_E13_500 = True
        use_aod_cimel_500 = True
 
    # -------------------------------------------------------------
    # Compare aod_nimfr and aod_cimel
    # -------------------------------------------------------------
    dsproc.debug_lv1("Comparing aod_nimfr and aod_cimel")
    daily_index = 5

    good_data =  good_nimfr_500_index * good_cimel_500_index

    good_aod_nimfr_500 = aod_nimfr_500[good_data]
    good_aod_cimel_500 = aod_cimel_500[good_data]

    mean_bias, r2, slope, npoints, good_compare = compare_aod_statistics(good_aod_nimfr_500, good_aod_cimel_500,
                                                                         has_cimel = True)

    daily_RMSD_500[daily_index]  = r2
    daily_npoint_500[daily_index]  = npoints
    daily_mean_bias_500[daily_index]  = mean_bias
    daily_slope_500[daily_index] = slope

    if good_compare:
        use_aod_nimfr_500 = True
        use_aod_cimel_500 = True

    # -------------------------------------------------------------
    # Set good data flag
    # -------------------------------------------------------------

    # Have check here for whether is a good/bad day for nimfr data
    if bad_nimfr_days and begin_day in bad_nimfr_days:
        string = "Input day {} in list of bad NIMFR days, excluding NIMFR data from AOD best estimate calculations".format(begin_day)
        dsproc.debug_lv1(string)
        use_aod_nimfr_500 = False

    if use_aod_mfrsr_C1_500:
        good_data_flag += 1
    if use_aod_mfrsr_E13_500:
        good_data_flag += 2
    if use_aod_nimfr_500: 
        good_data_flag += 4
    if use_aod_cimel_500:
        good_data_flag += 8


    # -------------------------------------------------------------
    # Calculate AOD best estimate
    # -------------------------------------------------------------

    # AOD 500 nm variables
    nsamples = len(aod_cimel_500)

    aod_be_500 = init_var_data(out_ds, "aod_be_500", 0, nsamples, use_missing=True)
    if aod_be_500 is None:
        return -1

    qc_aod_be_500 = init_var_data(out_ds, "qc_aod_be_500", 0, nsamples )
    if qc_aod_be_500 is None:
        return -1

    aod_be_500_rand_unc = init_var_data(out_ds, "aod_be_500_random_uncertainty", 0, nsamples, use_missing=True)
    if aod_be_500_rand_unc is None:
        return -1

    aod_be_500_quad_unc = init_var_data(out_ds, "aod_be_500_quadrature_uncertainty", 0, nsamples, use_missing=True)
    if aod_be_500_quad_unc is None:
        return -1

    aod_be_500_source = init_var_data(out_ds, "aod_be_500_source", 0, nsamples, use_missing=True)
    if aod_be_500_source is None:
        return -1

    aod_be_500_range = init_var_data(out_ds, "aod_be_500_range", 0, nsamples, use_missing=True)
    if aod_be_500_range is None:
        return -1

    # AOD 870 nm variables
    aod_be_870 = init_var_data(out_ds, "aod_be_870", 0, nsamples, use_missing=True)
    if aod_be_870 is None:
        return -1

    qc_aod_be_870 = init_var_data(out_ds, "qc_aod_be_870", 0, nsamples )
    if qc_aod_be_870 is None:
        return -1

    aod_be_870_rand_unc = init_var_data(out_ds, "aod_be_870_random_uncertainty", 0, nsamples, use_missing=True)
    if aod_be_870_rand_unc is None:
        return -1

    aod_be_870_quad_unc = init_var_data(out_ds, "aod_be_870_quadrature_uncertainty", 0, nsamples, use_missing=True)
    if aod_be_870_quad_unc is None:
        return -1

    aod_be_870_source = init_var_data(out_ds, "aod_be_870_source", 0, nsamples, use_missing=True)
    if aod_be_870_source is None:
        return -1

    aod_be_870_range = init_var_data(out_ds, "aod_be_870_range", 0, nsamples, use_missing=True)
    if aod_be_870_range is None:
        return -1



    QC_INDETERMINATE = 1
    QC_BAD = 2

    for i in range(nsamples):
        source_flag_500 = 0
        source_flag_870 = 0
        measures_500 = []  
        measures_870 = []  
        
        # 500 nm
        if use_aod_mfrsr_C1_500 and good_mfrsr_C1_500_index[i] :
            source_flag_500 += 1
            measures_500.append(aod_mfrsr_C1_500[i])
            
        if use_aod_mfrsr_E13_500 and good_mfrsr_E13_500_index[i]:
            source_flag_500 += 2
            measures_500.append(aod_mfrsr_E13_500[i])

        if use_aod_nimfr_500 and good_nimfr_500_index[i]: 
            source_flag_500 += 4
            measures_500.append(aod_nimfr_500[i])

        if use_aod_cimel_500 and good_cimel_500_index[i]:
            source_flag_500 += 8
            measures_500.append(aod_cimel_500[i])

        nmeasures_500 = len(measures_500)

        # If no good data, mark as missing value
        if not measures_500:
            aod_be_500[i] = MISSING_VALUE
            qc_aod_be_500[i] = QC_BAD
            aod_be_500_rand_unc[i] = MISSING_VALUE

            aod_be_500_range[i] = MISSING_VALUE
            aod_be_500_quad_unc[i] = MISSING_VALUE
        # Only one measure, mark as indeterminate
        # Use standard uncertainty of 0.02
        elif nmeasures_500 == 1:
            aod_be_500[i] = measures_500[0]
            qc_aod_be_500[i] = QC_INDETERMINATE
            aod_be_500_rand_unc[i] = 0.02

            aod_be_500_range[i] = 0
            aod_be_500_quad_unc[i] = 0.02
        else:
            aod_be_500[i] = np.mean(measures_500)
            aod_be_500_rand_unc[i] = np.std(measures_500, ddof=1)
            qc_aod_be_500[i] = 0

            aod_be_500_range[i] = np.max(measures_500) - np.min(measures_500)
            aod_be_500_quad_unc[i] = np.sqrt( np.power(0.02, 2)*nmeasures_500 )


        aod_be_500_source[i] = source_flag_500


        # 870 nm
        if use_aod_mfrsr_C1_500 and good_mfrsr_C1_870_index[i] :
            source_flag_870 += 1
            measures_870.append(aod_mfrsr_C1_870[i])
            
        if use_aod_mfrsr_E13_500 and good_mfrsr_E13_870_index[i]:
            source_flag_870 += 2
            measures_870.append(aod_mfrsr_E13_870[i])

        if use_aod_nimfr_500 and good_nimfr_870_index[i]: 
            source_flag_870 += 4
            measures_870.append(aod_nimfr_870[i])

        if use_aod_cimel_500 and good_cimel_870_index[i]:
            source_flag_870 += 8
            measures_870.append(aod_cimel_870[i])

        nmeasures_870 = len(measures_870)

        # If no good data, mark as missing value
        if not measures_870:
            aod_be_870[i] = MISSING_VALUE
            qc_aod_be_870[i] = QC_BAD
            aod_be_870_rand_unc[i] = MISSING_VALUE

            aod_be_870_range[i] = MISSING_VALUE
            aod_be_870_quad_unc[i] = MISSING_VALUE
        # Only one measure, mark as indeterminate
        # Use standard uncertainty of 0.02
        elif nmeasures_870 == 1:
            aod_be_870[i] = measures_870[0]
            qc_aod_be_870[i] = QC_INDETERMINATE
            aod_be_870_rand_unc[i] = 0.02

            aod_be_870_range[i] = 0
            aod_be_870_quad_unc[i] = 0.02
        else:
            aod_be_870[i] = np.mean(measures_870)
            aod_be_870_rand_unc[i] = np.std(measures_870, ddof=1)
            qc_aod_be_870[i] = 0

            aod_be_870_range[i] = np.max(measures_870) - np.min(measures_870)
            aod_be_870_quad_unc[i] = np.sqrt( np.power(0.02, 2)*nmeasures_870 )

        aod_be_870_source[i] = source_flag_870



    # -------------------------------------------------------------
    # Store data
    # -------------------------------------------------------------

    # Store ncomparison data
    ncomparisons = np.array([i for i in range(6)], dtype=np.int32)

    out_var = dsproc.get_var(out_ds, "ncomparisons")
    if out_var is None:
        return -1  

    status = dsproc.set_var_data(out_var, 0, 6, MISSING_VALUE, ncomparisons)
    if status is None:
        return -1  
    

    # Store good_flag_data
    gdf = np.array([good_data_flag], dtype=np.int32)

    out_var = dsproc.get_var(out_ds, "good_data_flag")
    if out_var is None:
        return -1  

    status = dsproc.set_var_data(out_var, 0, 1, MISSING_VALUE, gdf)
    if status is None:
        return -1  


    # -------------------------------------------------------------
    # End algorithm
    # -------------------------------------------------------------

    if dsproc.get_debug_level() > 1:
        dsproc.dump_output_datasets("./debug_dumps", "output_data.debug", 0)

    return 1


# Quicklook hook function 
#
# This function will be called once per processing interval
# just after data is stored after process_data is complete.
#
#  @param  user_data - void pointer to the UserData structure
#                      returned by the init_process() function
#  @param  begin_date - begin time of the processing interval
#  @param  end_date   - end time of the processing interval
#      
#  @return
#    -  1 if processing should continue normally
#    -  0 if processing should skip the current processing interval
#         and continue on to the next one.
#    - -1 if a fatal error occurred and the process should exit.
def create_quicklook(proc_data, begin_date, end_date):

    status_string = 'create_quicklook error'

    out_dsid = proc_data.out_dsid

    # **************************
    # Create plots using run_dq_inpsector function
    # **************************
    dq_inputs = ['-t','-v','all','-E','time_offset','time',
                 'ncomparisons','time_bounds', None]

    status = dsproc.run_dq_inspector(out_dsid, begin_date, end_date, dq_inputs, 0)
    if status < 0:
        error_string = 'Could not execute call to dq_inspector' 
        dsproc.error(status_string, error_string)
        return -1

    return 1 


#  Main entry function.

#  @param  argc - number of command line arguments
#  @param  argv - command line arguments
#  @return
#    - 0 if successful
#    - 1 if an error occurred

def main():
    import sys
    proc_names = [ "qc_aod" ]
    dsproc.use_nc_extension()


    dsproc.set_init_process_hook(init_process)
    dsproc.set_process_data_hook(process_data)
    dsproc.set_finish_process_hook(finish_process)

    dsproc.set_pre_transform_hook(pre_transform_hook)
    dsproc.set_post_transform_hook(post_transform_hook)

    # Commenting quicklook, will not use in production 
    dsproc.set_quicklook_hook(create_quicklook)

    exit_value = dsproc.main(
        sys.argv,
        dsproc.PM_TRANSFORM_VAP,
        __version__,
        proc_names)

    return exit_value

if __name__ == '__main__':
    sys.exit(main())
               
