from datetime import datetime, timezone, date, timedelta, time
from astropy.io import fits 
import numpy as np
import matplotlib.pyplot as plt 
plt.ion() 
import matplotlib
from glob import glob
import os 
from astroplan import Observer
from astropy.time import Time 
from matplotlib.patches import Rectangle 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u 
import pickle 

def on_sky_metric_plot(year=None, live_plot=False):
    # initialize astroplan Observer object so we can calculate night durations
    tierras = Observer.at_site('whipple', timezone='US/Arizona')

    # get the time of now. 
    current_utc_time = datetime.now(timezone.utc)
    current_year = current_utc_time.year
    
    # if a year hasn't been passed, take it to be the current year. 
    if year is None:
        year = current_year

    # create arrays that will contain the relative and absolute on-sky metrics.
    # represent as 7 days x 53 weeks array, which covers the year's duration in both normal and leap years.
    # "relative" =  total exposure time / night duration (astronomical evening twlight to astronomical morning twlight).
    # absolute = total exposure time in hours.
    # initialize everything to nan's, as we won't want to visualize nights that haven't occurred yet!
    on_sky_relative = np.zeros((7, 53))+np.nan
    on_sky_absolute = np.zeros((7, 53))+np.nan


    incoming_dir = '/data/tierras/incoming/'

    # start from January 1st. 
    start_date = date.fromisoformat(f'{year}0101')

    # start_date = date.fromisoformat(f'{year}1006') #TESTING!!!

    # if we're looking at the current year, just do up to the current date
    if year == current_year:
        today = date.fromisoformat(current_utc_time.isoformat().split('T')[0].replace('-','')) + timedelta(days=1)
    else:
        # if we're looking at a previous year, do the full year
        today = date.fromisoformat(f'{year+1}0101')

    current_date = start_date
    i = 0 
    week = 0 

    fig, ax = plt.subplots(2,1,figsize=(12,4.5), sharex=True, sharey=True)
    fig.suptitle(year, fontsize=16)

    # set up a custom discrete colormap where "no data" corresponds to a value between -0.2 and 0, and has a color of grey 
    cmap = matplotlib.cm.viridis
    cmap.set_bad('white',1.)
    cmap_list = [(0.5, 0.5, 0.5), *cmap(np.linspace(0, 1, 256))] #
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_viridis', cmap_list, N=257) #
    
    # set the bounds for the relative colormap
    bounds1 = np.array([-0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0])
    norm1 = matplotlib.colors.BoundaryNorm(bounds1, custom_cmap.N)

    # set the bounds for the absolute colormap in hours
    bounds2 = np.arange(-2, 12, 2) 
    norm2 = matplotlib.colors.BoundaryNorm(bounds2, custom_cmap.N)

    # if there is existing on sky metric data, read it in and set the first date that is a NaN in the arrays
    if os.path.exists(f'/data/tierras/analytics/{year}_on_sky_metrics.p'): 
        on_sky_relative, on_sky_absolute, rects1, rects2, durations = pickle.load(open(f'/data/tierras/analytics/{year}_on_sky_metrics.p', 'rb'))
        try:
            first_nan_doy = np.where(np.isnan(np.ravel(on_sky_absolute.T)))[0][0]
            current_date = date(year, 1, 1) + timedelta(days=int(first_nan_doy))
        except:
            pass
    else:
        rects1 = []
        rects2 = []
        durations = []

    while current_date < today: 
        # calculate this night's duration.
        datestr = str(current_date.year)+str(current_date.month).zfill(2) + str(current_date.day).zfill(2)
        print('Doing '+datestr)
        date_observed = False
        time_ = datetime.combine(current_date, time(16,0,0))
        tonight = tierras.tonight(Time(time_, format='datetime'), horizon=-12*u.deg) # this should give astronomical twilight start/end times
        start = tonight[0].value
        stop = tonight[1].value 

        if stop - start < 0:
            breakpoint()
            # unclear why this bug happens, but specifying "next" seems to fix it
            start = tierras.twilight_evening_astronomical(Time(time_), which='next', n_grid_points=300).value
            stop = tierras.twilight_morning_astronomical(Time(time_), which='next', n_grid_points=300).value
        duration = (stop - start) * 24 * 60 * 60
        durations.append(duration)

        if abs(duration) > 60000:
            print('Wtf is happening')
            breakpoint()

        # if we took data on this night...
        if os.path.exists(incoming_dir+datestr):
            date_observed = True

            # ...sum the total exposure time from all the fits files on this night to figure out the relative and absolute time on sky
            fits_files = sorted(glob(incoming_dir+datestr+'/*.fit'))
            total_exp_time = 0
            for j in range(len(fits_files)): 
                if ('flat' not in fits_files[j].lower()) and ('target' not in fits_files[j].lower()) and ('test' not in fits_files[j].lower()) and ('vega' not in fits_files[j].lower()) and ('deneb' not in fits_files[j].lower()) and ('bias' not in fits_files[j].lower()):
                    try:
                        exptime = fits.open(fits_files[j])[0].header['EXPTIME']
                    except:
                        print(f'{fits_files[j]} corrupted, skipping!')
                        continue 
                    total_exp_time += exptime

        # advance the loop, incrementing the week counter if i%7 ==0.
        current_date += timedelta(days=1)
        if i % 7 == 0 and i != 0:
            week += 1

        # plot a border around each pixel for visualization.
        # save them to lists so they can be read in subsequent loops
        rects1.append(Rectangle((week - 0.5, i%7 - 0.5), 1, 1,
                         facecolor='none',  # No fill color
                         edgecolor='white', # Border color
                         linewidth=1.5)       # Border thickness
                    )
        
        rects2.append(Rectangle((week - 0.5, i%7 - 0.5), 1, 1,
                         facecolor='none',  # No fill color
                         edgecolor='white', # Border color
                         linewidth=1.5)       # Border thickness
                    )

        if live_plot:
            for r in range(len(rects1)):
                ax[0].add_patch(rects1[r])
                ax[1].add_patch(rects2[r])

        if date_observed:
            on_sky_relative[i % 7, week] = total_exp_time / duration
            on_sky_absolute[i % 7, week] = total_exp_time / 3600
        else:
            # bit of a hack: set nights with *no* data equal to -0.1, which will trigger the colormap to color them as grey in both the relative and absolute plots. 
            # we'll revert them to 0 after the plotting loop, this is just for visualization.
            on_sky_relative[i % 7, week] = -0.1 
            on_sky_absolute[i % 7, week] = -0.1 

        # convert to masked array so nights that haven't occurred yet are just shown in white
        masked_on_sky_relative  = np.ma.array(on_sky_relative, mask=np.isnan(on_sky_relative))
        masked_on_sky_absolute = np.ma.array(on_sky_absolute, mask=np.isnan(on_sky_absolute)) 
        
        if live_plot:
            a1 = ax[0].imshow(masked_on_sky_relative, interpolation='none', cmap=custom_cmap, norm=norm1)
            a2 = ax[1].imshow(masked_on_sky_absolute, interpolation='none', cmap=custom_cmap, norm=norm2)

            # add color bars 
            divider1 = make_axes_locatable(ax[0])
            cax1 = divider1.append_axes("right", size="2%", pad=0.1)
            cb1 = fig.colorbar(a1, norm=norm1, cax=cax1)
            cb1.ax.tick_params(labelsize=12) 
            labels1 =  cb1.ax.get_yticklabels()
            labels1[0] = matplotlib.text.Text(1, -0.2, 'No Data')
            cb1.ax.set_yticklabels(labels1) 
            cb1.set_label('Fraction of Night', fontsize=14)

            divider2 = make_axes_locatable(ax[1])
            cax2 = divider2.append_axes("right", size="2%", pad=0.1)
            cb2 = fig.colorbar(a2, norm=norm2, cax=cax2)
            cb2.ax.tick_params(labelsize=12) 
            labels2 =  cb2.ax.get_yticklabels()
            labels2[0] = matplotlib.text.Text(1, -2, 'No Data')
            cb2.ax.set_yticklabels(labels2)
            cb2.set_label('Time (hours)', fontsize=14)
            ax[0].tick_params(labelsize=12)
            ax[1].tick_params(labelsize=12)
            ax[0].set_yticks([])
            ax[1].set_yticks([])
            ax[1].set_xticks([0, 4, 8, 12, 17, 21, 26, 30, 34, 39, 43, 47], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            fig.tight_layout()

            plt.pause(0.1)  
            cb1.remove()
            cb2.remove()
            ax[0].cla()
            ax[1].cla()
                    
        i += 1
    
    on_sky_relative[np.where(on_sky_relative == 0)] = -0.1 # NaN any 0's that leaked through
    on_sky_absolute[np.where(on_sky_absolute == 0)] = -0.1

    # plot again now that the loop is complete 
    masked_on_sky_relative  = np.ma.array(on_sky_relative, mask=np.isnan(on_sky_relative))
    masked_on_sky_absolute = np.ma.array(on_sky_absolute, mask=np.isnan(on_sky_absolute)) 
    
    a1 = ax[0].imshow(masked_on_sky_relative, interpolation='none', cmap=custom_cmap, norm=norm1)

    a2 = ax[1].imshow(masked_on_sky_absolute, interpolation='none', cmap=custom_cmap, norm=norm2)

    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="2%", pad=0.1)
    cb1 = fig.colorbar(a1, norm=norm1, cax=cax1)
    cb1.ax.tick_params(labelsize=12) 
    labels1 =  cb1.ax.get_yticklabels()
    labels1[0] = matplotlib.text.Text(1, -0.2, 'No Data')
    cb1.ax.set_yticklabels(labels1) 
    cb1.set_label('Fraction of Night', fontsize=14)

    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="2%", pad=0.1)
    cb2 = fig.colorbar(a2, norm=norm2, cax=cax2)
    cb2.ax.tick_params(labelsize=12) 
    labels2 =  cb2.ax.get_yticklabels()
    labels2[0] = matplotlib.text.Text(1, -2, 'No Data')
    cb2.ax.set_yticklabels(labels2)
    cb2.set_label('Time (hours)', fontsize=14)

    # for r in range(len(rects1)):
    #     ax[0].add_patch(rects1[r])
    #     ax[1].add_patch(rects2[r])

    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(labelsize=12)
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[1].set_xticks([0, 4, 8, 12, 17, 21, 26, 30, 34, 39, 43, 47], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    fig.tight_layout()

    

    # fig.savefig(f'/data/tierras/analytics/{year}_on_sky_metrics.png', dpi=300)

    # do a plot of just the absolute time 
    fig, ax = plt.subplots(1,1,figsize=(12,2.))
    a3 = ax.imshow(masked_on_sky_absolute, interpolation='none', cmap=custom_cmap, norm=norm2)

    divider3 = make_axes_locatable(ax)
    cax3 = divider3.append_axes("right", size="2%", pad=0.1)
    cb3 = fig.colorbar(a3, norm=norm2, cax=cax3)
    cb3.ax.tick_params(labelsize=12) 
    labels3 =  cb3.ax.get_yticklabels()
    labels3[0] = matplotlib.text.Text(1, -2, 'No Data')
    cb3.ax.set_yticklabels(labels3)
    cb3.set_label('Time (hours)', fontsize=14)

    # add rects 
    current_date = date.fromisoformat(f'{year}0101')

    rects2 = []
    while current_date < today:
        current_date += timedelta(days=1)
        if i % 7 == 0 and i != 0:
            week += 1

        rects2.append(Rectangle((week - 0.5, i%7 - 0.5), 1, 1,
                         facecolor='none',  # No fill color
                         edgecolor='white', # Border color
                         linewidth=1.5)       # Border thickness
                    )    
        i += 1
    for r in range(len(rects2)):
        ax.add_patch(rects2[r])

    ax.tick_params(labelsize=12)
    ax.set_yticks([])
    ax.set_xticks([0, 4, 8, 12, 17, 21, 26, 30, 34, 39, 43, 47], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    fig.tight_layout()
    fig.savefig(f'/data/tierras/analytics/{year}_on_sky_absolute.png', dpi=300)


    # now that things have been plotted, update the arrays to have 0s on nights with no data instead of -0.1
    on_sky_relative[np.where(on_sky_relative == -0.1)] = 0.
    on_sky_absolute[np.where(on_sky_absolute == -0.1)] = 0.
    breakpoint()

    breakpoint()
    plt.close('all')

    pickle.dump((on_sky_relative, on_sky_absolute, rects1, rects2, durations), open(f'/data/tierras/analytics/{year}_on_sky_metrics.p', 'wb'))

    
if __name__ == '__main__':
    on_sky_metric_plot(year=2024, live_plot=True)
