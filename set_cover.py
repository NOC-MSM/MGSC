import xarray as xr
import numpy as np
import scipy.stats as st
from scipy.optimize import curve_fit
import scipy.ndimage.filters as filters
import coast.general_utils as gu
import matplotlib.pyplot as plt
import numpy.random as random
from numpy.random import default_rng

def effective_coverage_rate(c, thresh, locs, landmask, coverage = 1):
    
    c_thresh = c>thresh
    n_pts = c_thresh.shape[0]
    c_locs = c_thresh[locs]
    ecr = np.sum(c_locs, axis=0)
    ecr_map_all = np.copy(ecr)
    ecr[ecr<coverage] = 0
    ecr[ecr>=coverage] = 1
    ecr_map = np.copy(ecr)
    ecr = (np.sum(ecr))/(n_pts - np.sum(landmask))
    
    return ecr, ecr_map, ecr_map_all

def create_geo_subplots(lonbounds, latbounds, n_r=1, n_c=1, figsize=(7,7)):
    """
    A routine for creating an axis for any geographical plot. Within the
    specified longitude and latitude bounds, a map will be drawn up using
    cartopy. Any type of matplotlib plot can then be added to this figure.
    For example:
    Example Useage
    #############
        f,a = create_geo_axes(lonbounds, latbounds)
        sca = a.scatter(stats.longitude, stats.latitude, c=stats.corr,
                        vmin=.75, vmax=1,
                        edgecolors='k', linewidths=.5, zorder=100)
        f.colorbar(sca)
        a.set_title('SSH correlations \n Monthly PSMSL tide gauge vs CO9_AMM15p0',
                    fontsize=9)
    * Note: For scatter plots, it is useful to set zorder = 100 (or similar
            positive number)
    """

    import cartopy.crs as ccrs  # mapping plots
    from cartopy.feature import NaturalEarthFeature

    # If no figure or ax is provided, create a new one
    #fig = plt.figure()
    #fig.clf()
    fig, ax = plt.subplots(n_r, n_c, subplot_kw={'projection': ccrs.PlateCarree()},
                           sharey = True, sharex = True, figsize=figsize)
    ax = ax.flatten()

    for rr in range(n_r*n_c):
        coast = NaturalEarthFeature(category="physical", facecolor=[0.9, 0.9, 0.9], name="coastline", scale="50m")
        ax[rr].add_feature(coast, edgecolor="gray")
        # ax.coastlines(facecolor=[0.8,0.8,0.8])
        gl = ax[rr].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", linestyle="-")
        gl.top_labels = False
        gl.right_labels = False
        
        if rr%n_c==0:
            gl.left_labels = True
        else:
            gl.left_labels = False
            
        if np.abs(n_r*n_c - rr)<=n_c:
            gl.bottom_labels = True
        else:
            gl.bottom_labels = False
            
    
        ax[rr].set_xlim(lonbounds[0], lonbounds[1])
        ax[rr].set_ylim(latbounds[0], latbounds[1])
        ax[rr].set_aspect("auto")
    
    ax = ax.reshape((n_r, n_c))

    plt.show()
    return fig, ax

def set_cover_greedy_search(c, n_obs, search_start = 0.1, search_end = 1,
                            search_step = 0.1, max_it=np.inf, stop_crit=1, 
                            print_it=True, tolerance = 0, loc0=None, 
                            stop_alpha = 1e-5):
    
    searching=True
    c0 = search_start
    c1 = search_start
    
    c_list = []
    n_pts_list = []
    exceeded_once = False
    while searching:
        c_list.append(c1)
        if print_it:
            print('Searching for c = {0}'.format(c1))
            print('({0}, {1})'.format(c0, c1))
        subsets = c>=c1
        loc, N = set_cover_greedy(subsets, print_it=False, loc0=loc0)#, 
                                  #max_it=n_obs + tolerance+1)
        
        if len(loc) > 0:
            ful = set_cover_get_final_sets(subsets, loc).astype(bool)
            removed, keep, loc, ful = reassign_correlations_delete_sets(ful, loc, 
                                                                        c, c1)
        n_pts = len(loc)
        n_pts_list.append(n_pts)
        if print_it:
            print('   Found {0} points'.format(n_pts))
              
        if np.abs(n_pts - n_obs) <= tolerance:
            if print_it:
                print('Tolerance zone reached, stopping iteration')
            searching = False
        elif n_pts > n_obs + tolerance:
            search_step = search_step / 2
            if search_step < stop_alpha:
                if print_it:
                    print('stop_alpha reached, stopping iteration')
                searching = False
            c1 = c0 + search_step
            exceeded_once = True
        else:
            c0 = np.copy(c1)
            if exceeded_once:
                search_step = search_step/2
                if search_step < stop_alpha:
                    if print_it:
                        print('stop_alpha reached, stopping iteration')
                        searching = False
            c1 = c1 + search_step
            
            
    return loc, N, np.array(c_list), np.array(n_pts_list)

def set_cover_greedy(subsets, max_it=np.inf, stop_crit=1, print_it=True, loc0=None):
    '''
    Basic greedy algorithm for set cover problem

    '''
    if print_it:
        print(' ')
        print(' >>>>>> GREEDY ALGORITHM FOR SET COVERING <<<<<<')
        print('')
        print('Size of universe: {0}'.format(subsets.shape[1]))
        print('Number of subsets: {0}'.format(subsets.shape[0]))

    loc = [] # Locations in ORIGINAL correlation array for optimal tiles
    N = [] # Number of points in each tile
    
    # Remaining points for each iteration
    n_subsets, n_pts = subsets.shape
    
    uncovered = np.ones(n_pts).astype(bool)
    remaining_subsets = np.ones(n_subsets).astype(bool)
    subset_indices = np.arange(n_subsets)
    
    # Remove initial locations in loc0
    if loc0 is not None:
        remaining_subsets[loc0] = 0
        loc = loc0
        for ii in range(len(loc0)):
            subset_opt = subsets[ loc0[ii] ]
            uncovered[subset_opt] = 0
    
    # Remove any empty sets
    subset_sum = np.sum(subsets, axis=1)
    zero_ind = subset_sum == 0
    remaining_subsets[zero_ind] = 0
    #print('Removed {0} empty subsets'.format(np.sum(zero_ind)))
    if print_it:
        print('There are {0} empty subsets'.format(np.sum(zero_ind)))
    
    # Remove any uncoverables
    subset_sum = np.sum(subsets, axis=0)
    zero_ind = subset_sum == 0
    uncovered[zero_ind] = 0
    #print('Removed {0} uncoverable points'.format(np.sum(zero_ind)))
    if print_it:
        print('There are {0} uncoverable points'.format(np.sum(zero_ind)))
    
    # While there are points remaining to analyse
    it=0
    subset_N=np.nan
    if print_it:
        print('')
    while np.sum(uncovered) > 0 and it<=max_it:
        
        if print_it:
            print('{2}. Uncovered: {0}   :::   Last N: {1}'.format(np.sum(uncovered), subset_N, it))
        
        # Extract remaining points from original array
        subsets_it = subsets[remaining_subsets]
        subsets_it = subsets_it[:, uncovered]
        #subsets_it = subsets[uncovered, :]
        #subsets_it = subsets[:, uncovered]
        remaining_subset_indices = subset_indices[remaining_subsets]
        
        # Get the sum of points in each tile
        subset_sum = np.sum(subsets_it, axis=1)
        
        # Identify which tile contains the most
        subset_max = np.argmax(subset_sum)
        subset_N = subset_sum[subset_max]
        
        if subset_N < stop_crit:
            break
        
        # Append values into output arrays
        subset_loc = remaining_subset_indices[subset_max]
        loc.append( subset_loc )
        N.append( subset_N )
        
        # Update remaining points by removing tile
        remaining_subsets[subset_loc] = 0
        
        # Get the optimal subset index
        subset_opt = subsets[ subset_loc ]
        uncovered[np.where(subset_opt)[0]] = 0
        
        #Update counters
        it = it+1
        
    return np.array(loc), np.array(N)

def set_cover_get_final_sets(subsets, loc):
    '''
    From set_cover output, get a boolean representation of the final sets
    Takes the form: (n_sets, n_universe)
    '''

    
    n_subsets = len(loc)
    n_pts = subsets.shape[1]
    final_sets = np.zeros((n_subsets, n_pts))
    subsets_loc = np.copy( subsets[loc] )
    
    for ii in range(n_subsets):
        subsets_tmp = np.copy(subsets_loc[ii]).astype(bool)
        final_sets[ii] = subsets_tmp
        subsets_loc[:, subsets_tmp] = 0
        
    return final_sets

def reconstruct_get_linear_fit(data, final_sets, loc):
    '''
    Get a set of linear fit coefficients of the form y = ax + b
    Output of form (n_sets, n_universe)
    '''
    data = data.T
    n_sets, n_pts = final_sets.shape
    linear_fits_a = np.zeros(final_sets.shape)*np.nan
    linear_fits_b = np.zeros(final_sets.shape)*np.nan
    
    def func(x, a, b):
        return a*x + b
    
    for ii in range(n_sets):
        print(ii)
        this_set = final_sets[ii]
        this_data = data[this_set]
        this_pts = np.sum(this_set)
        this_fit_a = np.zeros(this_pts)*np.nan
        this_fit_b = np.zeros(this_pts)*np.nan
        x = data[loc[ii]]
        for jj in range(this_pts):
            y = this_data[jj]
            fit_tmp,_ = curve_fit(func, x, y)
            this_fit_a[jj] = fit_tmp[0]
            this_fit_b[jj] = fit_tmp[1]
            
        linear_fits_a[ii, this_set] = this_fit_a
        linear_fits_b[ii, this_set] = this_fit_b
    return linear_fits_a, linear_fits_b

def reconstruct_linear(data_at_locs, linear_fits_a, linear_fits_b):
    
    n_sets, n_pts = linear_fits_a.shape
    reconstructed = np.zeros(n_pts)*np.nan
    
    def func(x, a, b):
        return a*x + b
    
    for ii in range(n_sets):
        print(ii)
        a_tmp = linear_fits_a[ii]
        b_tmp = linear_fits_b[ii]
        a_where = np.where( ~np.isnan(a_tmp) )[0]
        b_where = np.where( ~np.isnan(b_tmp) )[0]
        a_tmp = a_tmp[a_where]
        b_tmp = b_tmp[b_where]
        n_a = np.sum(~np.isnan(a_tmp))
        for jj in range(n_a):
            reconstructed[a_where[jj]] = func(data_at_locs[ii], a_tmp[jj], b_tmp[jj])
    return reconstructed

def compress_fullsets_indices(ful, loc):
    n_set, n_pts = ful.shape
    ful_ind = np.zeros(n_pts)*np.nan
    for ii in range(n_set):
        ful_ind[ful[ii]] = ii
    return ful_ind.astype(int)

def expand_fullsets_indices(ful_ind):
    n_pts = len(ful_ind)
    n_set = (np.nanmax(ful_ind) + 1).astype(int)
    ful_ex = np.zeros((n_set, n_pts))*np.nan
    for ii in range(n_set):
        ful_ex[ii] = ful_ind==ii
    return ful_ex.astype(bool)

def reassign_correlations_delete_sets(ful, loc, c, thresh, max_N = np.inf):
    
    # Create copy because we will assign
    ful_copy = np.copy(ful)
    n_set = ful_copy.shape[0]
    
    condition = True
    removed_sets = []
    keep_sets = np.arange(n_set, dtype=float)
    checked_sets = []
    while condition:
        
        ful_N = np.sum(ful_copy, axis=1).astype(float)
        ful_N[checked_sets] = np.nan
        
        if np.nanmin(ful_N) > max_N:
            break
        
        cc = np.nanargmin(ful_N)
        
        ful_tmp = ful_copy[ cc ]
        n_set_pts = ful_N[cc].astype(int)
        ful_ind = np.where(ful_tmp)[0]
        
        pt_move = np.zeros(n_set_pts).astype(int)
        pt_move_to = np.zeros(n_set_pts).astype(int)
        
        for ii in range(n_set_pts):
            
            c_tmp = np.zeros(n_set)*np.nan
            #x = data[ful_ind[ii]]
            for jj in range(n_set):
                #y = data[loc[jj]]
                #c_tmp[jj] = np.corrcoef(x,y)[0,1]
                c_tmp[jj] = c[ful_ind[ii], loc[jj]]
                
            c_tmp[cc] = np.nan
            c_tmp[removed_sets] = np.nan
                
            if np.sum(c_tmp>=thresh) >= 1:
                pt_move[ii] = True
                pt_move_to[ii] = np.nanargmax(c_tmp).astype(int)

        #print('{0} : {1}'.format( np.sum(np.isnan(ful_N)), np.nanmin(ful_N)))

        if all(pt_move):
            removed_sets.append(cc)
            ful_copy[pt_move_to, ful_ind] = True
            ful_copy[cc, ful_ind] = False
            keep_sets[cc] = np.nan
        checked_sets.append(cc)
        
        if len(checked_sets) == n_set:
            condition = False
            
    keep_sets = keep_sets[~np.isnan(keep_sets)]
    keep_sets = keep_sets.astype(int)
    new_loc = loc[keep_sets]
    ful_copy = ful_copy[keep_sets]
    
    return np.sort(removed_sets), keep_sets, new_loc, ful_copy

def reassign_correlations_improve(loc, ful, c):
    
    n_pts = c.shape[0]
    n_set = len(loc)
    new_ful_ind = np.zeros(n_pts)*np.nan
    
    for ii in range(n_pts):
        
        cor = np.zeros(n_set)*np.nan
        for ss in range(n_set):
            cor[ss] = c[ii, loc[ss]]
            new_ful_ind[ii] = np.argmax(cor)
        
    new_ful = expand_fullsets_indices(new_ful_ind)
    
    return new_ful

def construct_correlations(c, ful, loc):
    
    n_set, n_pts = ful.shape
    c2 = np.zeros(n_pts)*np.nan
    
    for ss in range(n_set):
        c2[ful[ss]] = c[loc[ss], ful[ss]]
    
    return c2

def monthly_anomalies_ostia(data):
    
    # Get Monthly Timeseries
    data_month = data.resample(time='1M').mean()
    data_month['time'] = data_month.time.values - np.timedelta64(6,'D')
    
    # Interpolate monthly back to original times
    data_month = data_month.interp(time=data.time)
    
    # Subtract monthly means from original data
    data_anom = data - data_month
    return data_anom

def calculate_haversine_distance(lon1, lat1, lon2, lat2):
    """
    # Estimation of geographical distance using the Haversine function.
    # Input can be single values or 1D arrays of locations. This
    # does NOT create a distance matrix but outputs another 1D array.
    # This works for either location vectors of equal length OR a single loc
    # and an arbitrary length location vector.
    #
    # lon1, lat1 :: Location(s) 1.
    # lon2, lat2 :: Location(s) 2.
    """

    # Convert to radians for calculations
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    # Latitude and longitude differences
    dlat = (lat2 - lat1) / 2
    dlon = (lon2 - lon1) / 2

    # Haversine function.
    distance = np.sin(dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon) ** 2
    distance = 2 * 6371.007176 * np.arcsin(np.sqrt(distance))

    return distance

def point_density_radial(grid_x, grid_y, loc_x, loc_y, r=100, 
                         adjust_land_proportion = False,
                         landmask = None,
                         input_flat = True, input_2D = False, 
                         input_simple = False):
    if input_simple:
        x2, y2 = np.meshgrid(grid_x,grid_y)
        x2F = x2.flatten()
        y2F = y2.flatten()
    if input_2D:
        x2F = grid_x.flatten()
        y2F = grid_y.flatten()
    if input_flat:
        x2F = grid_x
        y2F = grid_y
        
    n_p = len(x2F)
    n_l = len(loc_x)
    density = np.zeros(n_p)*np.nan
    density_adj = np.zeros(n_p)*np.nan
    lm_proportion = np.zeros(n_p)*np.nan
    area = np.pi*r**2
    
    # Loop over ALL points
    for pp in range(n_p):
        # Find all set_cover points within radius
        dist = calculate_haversine_distance(x2F[pp], y2F[pp], loc_x, loc_y)
        distr = dist<=r
        distrsum = np.nansum(distr)
        
        # Calculate first guess of density
        density[pp] = distrsum/area
        
        if adjust_land_proportion:
            search_box = r/111 + 1
            box_pts = np.logical_and(np.logical_and(x2F[pp]-search_box, x2F[pp] + search_box), 
                                     np.logical_and(y2F[pp]-search_box, y2F[pp] + search_box) )
            x2F_search = x2F[box_pts]
            y2F_search = y2F[box_pts]
            lm_search = landmask[box_pts]
            dist = calculate_haversine_distance(x2F[pp], y2F[pp], x2F_search, y2F_search)
            distr = dist<=r
            lm_radius = lm_search[distr]
            lm_proportion[pp] = np.sum(lm_radius)/len(lm_radius)
            density_adj[pp] = distrsum / ((1-lm_proportion[pp])*area)
    
    min_density = np.min( density[density>0] )
    density = np.clip(density, min_density, np.inf)
    
    if adjust_land_proportion:
        density[landmask] = np.nan
        density_adj[landmask] = np.nan
        lm_proportion[landmask ] = np.nan
        return density, density_adj, lm_proportion
    else:
        return density

def point_density_fractal():
    return

def point_density_reciprocal(N, ):
    return

def generate_loc_random(n_loc, lonF, latF, landmaskF):
    
    rng = default_rng()
    
    lm = landmaskF
    orig_indices = np.arange(len(lonF))

    lonF_oce = lonF[~lm]
    latF_oce = latF[~lm]
    orig_indices_oce = orig_indices[~lm]
    
    len_oce = len(lonF_oce)
    rand_ind = rng.choice(len_oce, size=n_loc, replace=False)
    rand_ind = orig_indices_oce[rand_ind]
    rand_lon = lonF[rand_ind]
    rand_lat = latF[rand_ind]
    
    return rand_ind, rand_lon, rand_lat

def generate_loc_regular(n_loc, lon2, lat2, landmask):
    
    ss0 = 2
    
    not_reached=True
    ss = ss0
    orig_indices = np.arange(len(lon2.flatten()))
    orig_indices2 = orig_indices.reshape(lon2.shape)
    while not_reached:
        lm_ss = landmask[::ss, ::ss]
        n_pts = np.sum(~lm_ss)
        if n_pts <= n_loc:
            not_reached = False
        else:
            n_pts_prev = n_pts
            ss = ss+1
        
    diff = np.abs(n_pts - n_loc)
    diff2 = np.abs(n_pts_prev - n_loc)
    
    if diff < diff2:
        ss_final = ss
    else:
        ss_final = ss-1

    lon2_ss = lon2[::ss_final, ::ss_final]
    lat2_ss = lat2[::ss_final, ::ss_final]
    lm_ss = landmask[::ss_final, ::ss_final]
    orig_indices2_ss = orig_indices2[::ss_final, ::ss_final]
    
    lon2_ssF = lon2_ss[~lm_ss]
    lat2_ssF = lat2_ss[~lm_ss]
    orig_indices2_ssF = orig_indices2_ss[~lm_ss]
    
    return orig_indices2_ssF, lon2_ssF, lat2_ssF

def fit_and_reconstruct_linear(loc, c, dataF_fit, dataF_recon, landmaskF):
    '''
    Do both the fitting and the reconstructing of a dataset, based on provided
    locations in loc.
    
    loc : Flattened location indices
    c   : Correlations matrix
    dataF_fit : Flattened data to use for fitting
    dataF_recon : Flattened data to use for reconstruction (e.g. differnt time)
    landmaskF : Flattened landmask to identify ignore points

    '''
    
    def fit_func(x, a, b):
        return a*x + b
    
    n_t, n_p = dataF_fit.shape
    c_pts = c[loc, :]
    c_pts_m = np.ma.masked_invalid(c_pts)
    c_max = np.nanmax(c_pts, axis=0)
    c_argmax = np.nanargmax(c_pts_m, axis=0)
    data_at_locs = dataF_fit[:,loc]

    param_a = np.zeros(len(c_argmax))*np.nan
    param_b = np.zeros(len(c_argmax))*np.nan
    data_recon = np.zeros(dataF_fit.shape)*np.nan

    # Get parameter fits
    for ii in range(len(c_argmax)):
        if landmaskF[ii]:
            continue
        y = dataF_fit[:,ii]
        x = data_at_locs[:, c_argmax[ii]]
        tmp, _ = curve_fit(fit_func, x, y)
        param_a[ii] = tmp[0]
        param_b[ii] = tmp[1]
        
    # Reconstruct data
    data_at_locs = dataF_recon[:,loc]
    for ii in range(len(c_argmax)):
        if landmaskF[ii]:
            continue
        x = data_at_locs[:, c_argmax[ii]]
        ts_rec = param_a[ii]*x + param_b[ii]
        data_recon[:, ii] = ts_rec
        
    return param_a, param_b, data_recon, c_max, c_argmax
