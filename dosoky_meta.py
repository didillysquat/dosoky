"""
This script is us starting again working within the context of being more meta.
This firstly means creating a new study that will contain the previously determined datasets
and then generating an output from this.
"""

from collections import defaultdict
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
from dbApp.models import DataSetSample, AnalysisType, CladeCollection, CladeCollectionType, Study, DataSet

import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.gridspec as gridspec
import os
import pickle
import itertools

import cartopy.crs as ccrs
import cartopy
from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE, OCEAN
from cartopy.mpl.gridliner import Gridliner
import math
import spcolors

class DosokyMeta:
    # The input attributes control the map area we are plotting and how the networks are plotted
    # area can be "rs", "rs_pag" or "global".
    # dosoky_centric = True or False. If True then we will only draw lines to and from the dosoky
    # site and not between the other sites.
    # clade can be A, C, D or all
    def __init__(self, plot_no_lat_lon=False, area="rs_pag", dosoky_centric=False, clade="C"):
        self.dosoky_centric = dosoky_centric
        self.clade = clade
        
        self.meta_df, self.post_med_count_df_abs, self.post_med_count_df_rel, self.profile_meta_df, self.profile_count_df_rel = self._read_in_symportal_objects()

        self._remove_bad_profiles_and_associated_samples()
        
        if not plot_no_lat_lon:
            self._remove_no_lat_lon_samples()

        self.names_of_data_sets_to_use = [
            "hume_et_al_2020", "20201018_schupp_all", "20190617_ziegler",
            "120418_buitrago_sp_v0.3.8", "20190612_dosoky", "240418_buitrago_sp_v0.3.8",
            "20210506_terraneo", "20210412_ziegler", "20171009_roberto_sp_v0.3.8", "terraneo_et_al_2019",
            "20170724_smith_singapore_v0_3_14",  "20190612_cardenas", "20190612_rossbach", "gardener_et_al_2019",
            "20170506_roberto_sp_v0.3.8", "20171017_roberto_sp_v0.3.8", "20171114_roberto_sp_v0.3.8"
        ]

        self.data_sets = [DataSet.objects.get(name=_) for _ in self.names_of_data_sets_to_use]

        self._remove_schupp_larval()

        # Sort the profile count df by abundance of the profiles
        self.profile_count_df_rel = self.profile_count_df_rel[self.profile_count_df_rel.sum(axis=0).sort_values(ascending=False).index.values]

        self.sample_uid_to_dataset_uid_dict, self.sample_uid_to_dataset_name_dict, self.data_set_uid_to_data_set_name_dict, self.dosoky_lat, self.dosoky_lon, self.dosoky_id, self.dosoky_dss_list = self._get_lat_lon_info()

        self.fig, self.ax, self.lat_lims, self.lon_lims = self._make_base_map(area=area)

        self.lat_lon_size_dosoky_ddict, self.lat_lon_size_no_lat_lon_ddict, self.lat_lon_size_other_ddict, self.dss_to_lat_lon_dict, self.sample_uid_no_lat_lon, self.no_lat_lon_ds_set = self._collect_site_plotting_data()

        self._remove_samples_outside_of_chosen_area()

        # If we are working as dosoky centric then we only want to conern ourselves
        # with those profiles that were found at the dososky sites.
        
        self.dosoky_profiles_list = self._get_dosoky_profiles_list()

    def _get_dosoky_profiles_list(self):
        dosoky_dss_list_still_good = [_ for _ in self.profile_count_df_rel.index if _ in [_.id for _ in self.dosoky_dss_list]]
        dosoky_profiles_df = self.profile_count_df_rel.loc[dosoky_dss_list_still_good,]
        dosoky_profiles_df = dosoky_profiles_df.loc[:,dosoky_profiles_df.any(axis=0)]
        return list(dosoky_profiles_df)

    def _remove_samples_outside_of_chosen_area(self):
        # We have to control which samples are plotted up either
        # at the point of plotting, or by removing them from
        # the count tables.
        # We will do the latter here.
        # we have the self.lat_lims and self.lon_lims and we can exlude samples according to these
        dss_to_cut = []
        for dss, lat_lon in self.dss_to_lat_lon_dict.items():
            # Check if lat within lat lim and lon within lon lim
            if self.lat_lims[0] < lat_lon[0] < self.lat_lims[1]:
                if self.lon_lims[0] < lat_lon[1] < self.lon_lims[1]:
                    continue
            dss_to_cut.append(dss)
        if dss_to_cut:
            to_keep = [_ for _ in self.meta_df.index if _ not in dss_to_cut]
            self.meta_df = self.meta_df.loc[to_keep,]
            self.profile_count_df_rel = self.profile_count_df_rel.loc[to_keep,]
            # Also need to remove the now empty profiles
            profiles_to_keep = self.profile_count_df_rel.any(axis=0)
            self.profile_count_df_rel = self.profile_count_df_rel.loc[:,profiles_to_keep]
            self.profile_meta_df = self.profile_meta_df.loc[profiles_to_keep,]
            self.post_med_count_df_abs = self.post_med_count_df_abs.loc[to_keep,]
            self.post_med_count_df_rel = self.post_med_count_df_rel.loc[to_keep,]
        else:
            # No samples to exclude
            return

    def _collect_site_plotting_data(self):
        # We want to plot up a scatter of the lat and lons
        # with size
        # I think it is helpful to have a seperate set of points for the dosoky, no lat long, and others
        lat_lon_size_dosoky_ddict = defaultdict(int)
        lat_lon_size_no_lat_lon_ddict = defaultdict(int)
        lat_lon_size_other_ddict = defaultdict(int)
        dss_to_lat_lon_dict = {}
        sample_uid_no_lat_lon = []
        no_lat_dd = defaultdict(int)
        for dss_uid, ser in self.meta_df.iterrows():
            if "999" in str(ser["collection_latitude"]):
                no_lat_dd[self.sample_uid_to_dataset_name_dict[dss_uid]] += 1
                print(f"No lat lon for {ser['sample_name']}")
                sample_uid_no_lat_lon.append(dss_uid)
                # Assiciate a lat long that puts them in the top right of the map for the time being
                lat_lon_size_no_lat_lon_ddict[",".join(["22", "90"])] += 1
                dss_to_lat_lon_dict[dss_uid] = (22,90)
            else:
                if self.sample_uid_to_dataset_uid_dict[dss_uid] == self.dosoky_id:
                    lat_lon_size_dosoky_ddict[",".join([str(ser["collection_latitude"]), str(ser["collection_longitude"])])] += 1
                else:
                    lat_lon_size_other_ddict[",".join([str(ser["collection_latitude"]), str(ser["collection_longitude"])])] += 1
                dss_to_lat_lon_dict[dss_uid] = (ser["collection_latitude"], ser["collection_longitude"])
        
        # Get the data set names of samples with no lat lon
        no_lat_lon_ds_set = {self.data_set_uid_to_data_set_name_dict[self.sample_uid_to_dataset_uid_dict[_]] for _ in sample_uid_no_lat_lon}
        return lat_lon_size_dosoky_ddict, lat_lon_size_no_lat_lon_ddict, lat_lon_size_other_ddict, dss_to_lat_lon_dict, sample_uid_no_lat_lon, no_lat_lon_ds_set

    def _make_base_map(self, area):
        # Make base plot of the sites
        # self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        # self.ax.set_xlabel("longitude Decimal Degree")
        # self.ax.set_ylabel("latitude Decimal Degree")
        fig = plt.figure(figsize=(15, 10))
        ax = plt.subplot(projection=ccrs.PlateCarree(), zorder=1)
        # land_110m, ocean_110m, boundary_110m = self._get_naural_earth_features_big_map()
        # print('Drawing annotations on map\n')
        # self._draw_natural_earth_features_big_map(land_110m, ocean_110m, boundary_110m)
        # print('Annotations complete\n')
        # Managed to download the ocean and land files from here: https://github.com/ingmapping/Basemaps_QTiles/tree/master/WorldMap/Data
        ax.add_feature(LAND, facecolor="white",
                                        edgecolor='black', linewidth=0.2)
        ax.add_feature(OCEAN, facecolor='#88b5e0',
                                        edgecolor='black', linewidth=0.2, alpha=0.3)
        if area == "global":
            lon_lims = (32, 150)
            lat_lims = (-6, 32)
        elif area == "rs":
            lon_lims = (32, 44)
            lat_lims = (12, 30)
        elif area == "rs_pag":
            lon_lims = (32, 60)
            lat_lims = (10, 32)
        else:
            raise RuntimeError("requested area not noticed")
        ax.set_extent(extents=(
                lon_lims[0], lon_lims[1], lat_lims[0], lat_lims[1]), crs=ccrs.PlateCarree())

        g1 = Gridliner(
                axes=ax, crs=ccrs.PlateCarree(), draw_labels=True)

        ax._gridliners.append(g1)
        # xlocs = mticker.FixedLocator([float(_) for _ in self.config_dict['lon_grid_line_pos'].split(',')])
        # ylocs = mticker.FixedLocator([float(_) for _ in self.config_dict['lat_grid_line_pos'].split(',')])
        # g1 = Gridliner(
        #     axes=self.large_map_ax, crs=ccrs.PlateCarree(), draw_labels=True,
        #     xlocator=xlocs, ylocator=ylocs)
        return fig, ax, lat_lims, lon_lims

    def _get_lat_lon_info(self):
        # make a dict that is sample_uid to dataset uid
        # Then append this information to the self.meta_df
        sample_uid_to_dataset_uid_dict = {}
        sample_uid_to_dataset_name_dict = {}
        data_set_uid_to_data_set_name_dict = {}
        for ds in self.data_sets:
            if ds.name == "20190612_dosoky":
                dosoky_id = ds.id
                dosoky_dss_list = list(DataSetSample.objects.filter(data_submission_from=ds))
            data_set_uid_to_data_set_name_dict[ds.id] = ds.name
            for dss in DataSetSample.objects.filter(data_submission_from=ds):
                # If it is one of the datasetsamples that we kicked out due to being with a dodgy
                # don't include it.
                if dss.id not in self.meta_df.index:
                    continue
                sample_uid_to_dataset_uid_dict[dss.id] = ds.id
                sample_uid_to_dataset_name_dict[dss.id] = ds.name
        
        self.meta_df["data_set_name"] = [sample_uid_to_dataset_name_dict[_] for _ in self.meta_df.index]
        self.meta_df["data_set_uid"] = [sample_uid_to_dataset_uid_dict[_] for _ in self.meta_df.index]
        dosoky_lat, dosoky_lon = 25.54, 34.64
        
        return sample_uid_to_dataset_uid_dict, sample_uid_to_dataset_name_dict, data_set_uid_to_data_set_name_dict, dosoky_lat, dosoky_lon, dosoky_id, dosoky_dss_list

    def _remove_schupp_larval(self):
        # We then also want to remove most of the samples from the schupp analysis
        # as these were experimental with larvae. But there are some adult colonies that
        # we have lat lon for and these are what we want to work with
        schupp_to_keep_uid_list = []
        schupp_to_drop_uid_list = []
        for ds in self.data_sets:
            if ds.name == "20201018_schupp_all":
                dss = DataSetSample.objects.filter(data_submission_from=ds).all()
                for d in dss:
                    if d.sample_type == "coral_field":
                        schupp_to_keep_uid_list.append(d.id)
                    else:
                        schupp_to_drop_uid_list.append(d.id)

        # Now drop these samples
        to_keep = [_ for _ in self.meta_df.index if _ not in schupp_to_drop_uid_list]
        self.meta_df = self.meta_df.loc[to_keep,]
        self.profile_count_df_rel = self.profile_count_df_rel.loc[to_keep,]
        self.post_med_count_df_abs = self.post_med_count_df_abs.loc[to_keep,]
        self.post_med_count_df_rel = self.post_med_count_df_rel.loc[to_keep,]

        # Now check to see if there were any profiles that were only found in the schupp samples that we removed
        # that now have no counts in them
        profiles_only_in_schupp = self.profile_count_df_rel.any(axis=0)
        print(f"{sum(profiles_only_in_schupp)} profiles were only found in the schupp larval samples")
        self.profile_count_df_rel = self.profile_count_df_rel.loc[:,self.profile_count_df_rel.any(axis=0)]
        print(f"{len(self.profile_count_df_rel.columns)} profiles left after removing")

    def _remove_no_lat_lon_samples(self):
        no_lat_lon_samples = []
        # Then we will remove those samples that have a lat long as 999
        for dss_uid, ser in self.meta_df.iterrows():
            if "999" in str(ser["collection_latitude"]):
                no_lat_lon_samples.append(dss_uid)
        # remove the samples and any now redunant profiles from the various dataframes
        print(f"removing {len(no_lat_lon_samples)} samples due to no lat lon")
        self.profile_count_df_rel = self.profile_count_df_rel.loc[[_ for _ in self.profile_count_df_rel.index if _ not in no_lat_lon_samples],]
        profs_to_keep = self.profile_count_df_rel.any(axis=0)
        print(f"dropping {len(self.profile_count_df_rel.columns)-sum(profs_to_keep)} now reduntant profiles")
        self.profile_count_df_rel = self.profile_count_df_rel.loc[:,profs_to_keep]
        self.meta_df = self.meta_df.loc[[_ for _ in self.meta_df.index if _ not in no_lat_lon_samples],]
        self.profile_meta_df = self.profile_meta_df.loc[[_ for _ in self.profile_meta_df.index if _ in list(self.profile_count_df_rel)],:]
        self.post_med_count_df_rel = self.post_med_count_df_rel.loc[[_ for _ in self.post_med_count_df_rel.index if _ not in no_lat_lon_samples],]
        self.post_med_count_df_abs = self.post_med_count_df_abs.loc[[_ for _ in self.post_med_count_df_abs.index if _ not in no_lat_lon_samples],]

    def _remove_bad_profiles_and_associated_samples(self):
        # Early on in this analysis we want to remove samples that contain profiles that are single DIV or only majabund e.g. A1/A1t or C1/C3/C1b
        # These samples are generally due to not finding good representative profiles
        # first find the profiles that match these criterea, then find the samples that contain them.
        # Then remove the profiles from the proile_meta_df and the profile count table,
        # then remove the samples from the profile_count_df_rel
        # First identify the profiles
        profile_to_del = []
        for profile_uid, ser in self.profile_meta_df.iterrows():
            # Work with the profile name to see if it is a profile we want to discard
            profile_name = ser["ITS2 type profile"]
            if "-" not in profile_name:
                profile_to_del.append(profile_uid)

        # del from the profile_meta_df
        profiles_to_keep = [_ for _ in self.profile_meta_df.index if _ not in profile_to_del]
        print(f"{len(profiles_to_keep)} profiles are being removed from the analysis due to being possibly aretefactual")
        self.profile_meta_df = self.profile_meta_df.loc[profiles_to_keep,]
        print(f"{len(self.profile_meta_df.index)} remain")
        
        # del samples and profiles from self.profile_count_df_rel
        # Slim down to only those types we want to delete
        to_del = self.profile_count_df_rel.loc[:, profile_to_del]
        # Get the samples that have values in any of these profiles
        to_del = to_del.any(axis=1)[to_del.any(axis=1) == True].index.values
        print(f"{len(to_del)} samples that are associated with these profiles are being removed from the analysis")
        # Convet this into the indices to keep
        index_keep = [_ for _ in self.profile_count_df_rel.index if _ not in to_del]
        header_keep = [_ for _ in list(self.profile_count_df_rel) if _ not in profile_to_del]
        print(f"{len(index_keep)} remain")
        # Remove the samples from the dataframes
        self.profile_count_df_rel = self.profile_count_df_rel.loc[index_keep, header_keep]
        self.meta_df = self.meta_df.loc[index_keep,]
        self.post_med_count_df_abs = self.post_med_count_df_abs.loc[index_keep,]
        self.post_med_count_df_rel = self.post_med_count_df_rel.loc[index_keep,]

    def _read_in_symportal_objects(self):
        if os.path.exists("cache/meta_df.p"):
            meta_df = pickle.load(open("cache/meta_df.p", "rb"))
        else:
            meta_df = pd.read_csv("20210906T062437/post_med_seqs/169_20210830_DBV_20210906T062437.seqs.absolute.meta_only.txt", sep="\t")
            meta_df.set_index("sample_uid", inplace=True, drop=True)
            pickle.dump(meta_df, open("cache/meta_df.p", "wb"))
        
        if os.path.exists("cache/post_med_count_df_abs.p"):
            post_med_count_df_abs = pickle.load(open("cache/post_med_count_df_abs.p", "rb"))
        else:
            post_med_count_df_abs = pd.read_csv("20210906T062437/post_med_seqs/169_20210830_DBV_20210906T062437.seqs.absolute.abund_only.txt", sep="\t")
            post_med_count_df_abs.set_index("sample_uid", inplace=True, drop=True)
            pickle.dump(post_med_count_df_abs, open("cache/post_med_count_df_abs.p", "wb"))
        
        if os.path.exists("cache/post_med_count_df_rel.p"):
            post_med_count_df_rel = pickle.load(open("cache/post_med_count_df_rel.p", "rb"))
        else:
            post_med_count_df_rel = post_med_count_df_abs.div(post_med_count_df_abs.sum(axis=1), axis=0)
            pickle.dump(post_med_count_df_rel, open("cache/post_med_count_df_rel.p", "wb"))
        
        if os.path.exists("cache/profile_meta_df.p"):
            profile_meta_df = pickle.load(open("cache/profile_meta_df.p", "rb"))
        else:
            profile_meta_df = pd.read_csv("20210906T062437/its2_type_profiles/169_20210830_DBV_20210906T062437.profiles.meta_only.txt", sep="\t")
            profile_meta_df.set_index("ITS2 type profile UID", inplace=True, drop=True)
            pickle.dump(profile_meta_df, open("cache/profile_meta_df.p", "wb"))
        profile_uid_to_profile_name_dict = {k:v for k, v in profile_meta_df["ITS2 type profile"].items()}

        if os.path.exists("cache/profile_count_df_rel.p"):
            profile_count_df_rel = pickle.load(open("cache/profile_count_df_rel.p", "rb"))
        else:
            profile_count_df_rel = pd.read_csv("20210906T062437/its2_type_profiles/169_20210830_DBV_20210906T062437.profiles.relative.abund_only.txt", sep="\t")
            profile_count_df_rel.set_index("sample_uid", inplace=True, drop=True)
            # make the headers int
            profile_count_df_rel.columns = [int(_) for _ in list(profile_count_df_rel)]
            pickle.dump(profile_count_df_rel, open("cache/profile_count_df_rel.p", "wb"))

        return meta_df, post_med_count_df_abs, post_med_count_df_rel, profile_meta_df, profile_count_df_rel
    
    @staticmethod
    def _get_naural_earth_features_big_map():
        land_110m = cartopy.feature.NaturalEarthFeature(category='physical', name='land',
                                                        scale='50m')
        ocean_110m = cartopy.feature.NaturalEarthFeature(category='physical', name='ocean',
                                                         scale='50m')
        boundary_110m = cartopy.feature.NaturalEarthFeature(category='cultural',
                                                            name='admin_0_boundary_lines_land', scale='110m')
        return land_110m, ocean_110m, boundary_110m

    def _draw_natural_earth_features_big_map(self, land_110m, ocean_110m, boundary_110m):
        """NB the RGB must be a tuple in a list and the R, G, B must be given as a value between 0 and 1"""
        # self.ax.add_feature(land_110m, facecolor=[(238 / 255, 239 / 255, 219 / 255)],
        #                               edgecolor='black', linewidth=0.2)
        
        self.ax.add_feature(land_110m, facecolor="white",
                                        edgecolor='black', linewidth=0.2, zorder=1)
        
        self.ax.add_feature(ocean_110m, facecolor='#88b5e0',
                                        edgecolor='black', linewidth=0.2)
        
        self.ax.add_feature(boundary_110m, edgecolor='gray', linewidth=0.2, facecolor='None')

    def _make_seq_color_dict(self, ref_seqs_from_profiles):
        seq_color_dict = {}
        for seq_name in ref_seqs_from_profiles:
            if seq_name in self.pre_def_seq_color_dict:
                seq_color_dict[seq_name] = self.pre_def_seq_color_dict[seq_name]
            else:
                try:
                    seq_color_dict[seq_name] = next(self.color_hash_iterator)
                except StopIteration:
                    seq_color_dict[seq_name] = next(self.grey_iterator)
        return seq_color_dict

    def start_bars_only(self):
        # here we will aim to plot up a figure for each of the profiles that
        # will have a main feature of the dosoky samples post-MED plotted
        # We will plot the samples grouped by lat lon pairing as distance fromthe dosoky site
        # We will then manually look at the produced plots and see which profiles are good
        # and which are bad.

        # Overwrite the current self.fig and self.ax to work with a gridspec setup

        
        # self.color_dict = self._make_seq_color_dict(self.post_med_count_df_rel.columns)

        for profile in self.profile_count_df_rel:
            # We are only interested in those profiles that were found in the dosoky
            if self.dosoky_centric:
                if profile not in self.dosoky_profiles_list:
                    continue
            # Get a list of the datasetsamples that have the profile
            # Then make a dict of lat_long to samples
            # Then make a lat_lon_to_dist dict
            # Then in order of the distance, plot up the groups of samples
            # For each group we'll also want the dataset name. We can go
            # from dss uid to ds uid to ds name.
            # We will have to do the region based on the lat lon.
            # if lon < 45 then rs
            # if lon > 140 then guam
            # if lon > 100 then singapore
            # if lat < 0 then seychelles 
            # else pag 
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(nrows=9, ncols=1)
            self.title_ax = plt.subplot(gs[0:1, :])
            self.bar_ax = plt.subplot(gs[1:4, :])
            self.leg_ax = plt.subplot(gs[4:5, :])
            self.dataset_ax = plt.subplot(gs[5:7, :])
            self.region_ax = plt.subplot(gs[7:8, :])
            self.species_ax = plt.subplot(gs[8:9, :])

            # First plot up the profile info
            profile_name = self.profile_meta_df.at[profile, "ITS2 type profile"]
            self.title_ax.text(s=f"{profile_name}; {profile}", x=0.5, y=0.5, va="center", ha="center")
            profile_clade = self.profile_meta_df.at[profile, "Clade"]

            # Now plot up the barcharts
            # Get a sub count table of the samples
            ser = self.profile_count_df_rel[profile]
            profile_dss = list(ser[ser != 0].index)
            ser = ser[profile_dss]

            # get lat_lon_to_dss_for_profile_dict
            lat_lon_to_dss_for_profile_dict = defaultdict(list)
            for dss, abund in ser.items():
                lat_lon_to_dss_for_profile_dict[self.dss_to_lat_lon_dict[dss]].append(dss)

            # get lat_lon_to_dist_to_dosoky_dict
            lat_lon_to_dist_to_dosoky_dict = {}
            for lat_lon in lat_lon_to_dss_for_profile_dict.keys():
                # get the distance eDistance = math.dist([Px, Py], [Qx, Qy])
                dist = math.dist([self.dosoky_lon, self.dosoky_lat], [lat_lon[1], lat_lon[0]])
                lat_lon_to_dist_to_dosoky_dict[lat_lon] = dist
            
            # Do an intial slim down of the count dict for the given profile.
            # We will then slim down further to the lat lon specific dds set
            clade_seqs = [_ for _ in list(self.post_med_count_df_rel) if _.startswith(profile_clade) or _.endswith(profile_clade)]
            count_df_to_plot = self.post_med_count_df_rel.loc[:,clade_seqs]
            seqs_with_counts_profile = count_df_to_plot.loc[profile_dss,].any(axis=0)
            count_df_to_plot_profile = count_df_to_plot.loc[profile_dss, seqs_with_counts_profile]
            # Make sure they are in order
            count_df_to_plot_profile = count_df_to_plot_profile[count_df_to_plot_profile.sum(axis=0).sort_values(ascending=False).index.values]
            # Make sure that the DIVs of the given profile are plotted in order
            div_list = profile_name.replace("/","_").replace("-","_").split("_")
            non_div_seqs = [_ for _ in count_df_to_plot_profile.columns if _ not in div_list]
            div_order = div_list + non_div_seqs
            count_df_to_plot_profile = count_df_to_plot_profile[div_order]
            # Setup the colour dict
            self.color_hash_iterator = iter(spcolors.color_list)
            self.grey_iterator = itertools.cycle(spcolors.greys)
            self.pre_def_seq_color_dict = spcolors.pre_def_color_dict
            self.color_dict = self._make_seq_color_dict(count_df_to_plot_profile.columns)

            # Plot up the seq legend 
            leg_bar_patches = []
            index_for_plot = 0
            for div in div_list:
                bottom = 0
                leg_bar_patches.append(Rectangle(
                            (index_for_plot, bottom),
                            0.5,
                            1, color=self.color_dict[div]))
                self.leg_ax.text(s=div, x=index_for_plot + .75, y=0.5, ha="left", va="center")
                index_for_plot += 2
            
            patches_collection = PatchCollection(leg_bar_patches, match_original=True)
            # patches_collection.set_array(np.arange(len(bar_patches)))
            self.leg_ax.add_collection(patches_collection)
            self.leg_ax.set_xlim(0, index_for_plot)

            # then plot up the samples in order of the distances
            index_for_plot = -0.5
            bar_patches = []
            color_list = []
            for lat_lon in [_[0] for _ in sorted(lat_lon_to_dist_to_dosoky_dict.items(), key=lambda x: x[1], reverse=False)]:
                index_for_plot += 0.5
                start_index = index_for_plot
                dss = lat_lon_to_dss_for_profile_dict[lat_lon]
                # slim down the post_med
                # Want to limit to those sequences of the same clade as the profile
                
                seqs_with_counts = count_df_to_plot_profile.loc[dss,].any(axis=0)
                count_df_to_plot = count_df_to_plot_profile.loc[dss, seqs_with_counts]
                
                # renormalise to 1
                count_df_to_plot = count_df_to_plot.div(count_df_to_plot.sum(axis=1), axis=0)
                for sample_uid in count_df_to_plot.index:
                    bottom = 0
                    non_zero_seq_abundances = count_df_to_plot.loc[sample_uid][
                        count_df_to_plot.loc[sample_uid] > 0]
                    
                    for obj_uid, abund in non_zero_seq_abundances.iteritems():
                        bar_patches.append(Rectangle(
                            (index_for_plot - 0.5, bottom),
                            1,
                            abund, color=self.color_dict[obj_uid]))
                        
                        bottom += abund
                        color_list.append(self.color_dict[obj_uid])
                    

                    # Add species info
                    genus = self.meta_df.at[sample_uid, "host_genus"]
                    species = self.meta_df.at[sample_uid, "host_species"]
                    if species == "NoData":
                        if genus == "NoData":
                            pass
                        else:
                            # genus with no species
                            self.species_ax.text(s=f"{genus} spp.", x=index_for_plot, y=0.5, va="center", ha="center", rotation="vertical")
                    else:
                        # Genus and species
                        self.species_ax.text(s=f"{genus[0]} {species}", x=index_for_plot, y=0.5, va="center", ha="center", rotation="vertical")

                    index_for_plot += 1
                
                # At this point we want to add the details of the dataset name and the region
                data_set_name = self.sample_uid_to_dataset_name_dict[sample_uid]
                self.dataset_ax.text(s=data_set_name, x=start_index + ((index_for_plot - start_index)/2) - 0.5, y=0.5, rotation="vertical", ha="center", va="center")
                
                # Add the region information to the plot
                if lat_lon[1] < 45:
                    region = "RS"
                elif lat_lon[1] > 140:
                    region = "Guam"
                elif lat_lon[1] > 100:
                    region = "Singapore"
                elif lat_lon[0] < 0:
                    region = "Seychelles"
                else:
                    region = "PAG"
                self.region_ax.text(s=region, x=start_index + ((index_for_plot - start_index)/2) - 0.5, y=0.5, rotation="vertical", ha="center", va="center")

            
            # Now render the bars on the plot
            listed_color_map = ListedColormap(color_list)
            patches_collection = PatchCollection(bar_patches, cmap=listed_color_map)
            patches_collection.set_array(np.arange(len(bar_patches)))
            self.bar_ax.add_collection(patches_collection)
            
            # adjust the lims of the dataset and region axes
            self.bar_ax.autoscale_view()
            self.region_ax.set_xlim(self.bar_ax.get_xlim())
            self.region_ax.set_ylim(0,1)
            self.dataset_ax.set_xlim(self.bar_ax.get_xlim())
            self.dataset_ax.set_ylim(0,1)
            self.species_ax.set_xlim(self.bar_ax.get_xlim())
            self.species_ax.set_ylim(0,1)

            for ax in [self.bar_ax, self.title_ax, self.leg_ax, self.region_ax, self.dataset_ax, self.species_ax]:
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

            plt.savefig(f"profile_plots/{profile}_{profile_clade}.png", dpi=1200)
            plt.close()

        foo = "bar"



    def start_map_only(self):
        # The main figure idea I had was to plot a map that shows the inter connectivity
        # between the sites according to the profiles that they have in common
        # To start with we can work in a Dosoky centric manner where we work through
        # the dosoky profiles and check to see where they are found.

        # After this we can work in a manner that is completely interconnected and plot up all
        # of the profile interactions for all samples.

        # Eventually this figure would be in the form of a map, but to start with,
        # to make things a little faster we can plot up just using the coordinates
        # ona regular plot.
        
        # self._plot_up_dosoky_centric()
        
        # At this point we have the connectivity on a dosoky centric way plotted.
        # The other approach that will be interesting will be to plot the interconnectivity between all profiles
        # The way to do this will be to go profile by profile
        # For each profile get the samples that have that profiles
        # Then make a count dict of the lat lons associated with those
        # Then we want to find all pairwise comparisions between those and plot a line to represent these.
        # That will be our network.

        # If we are working as dosoky centric then we only want to conern ourselves
        # with those profiles that were found at the dososky sites.
        

        all_lat_lons = set()
        profile_to_num_dss_lat_lon = {}
        for profile in self.profile_count_df_rel:
            if self.dosoky_centric:
                if profile not in self.dosoky_profiles_list:
                    continue
            clade = self.profile_meta_df.at[profile, "Clade"]
            if self.clade != "all":
                if clade != self.clade:
                    continue
            if clade == "C":
                color = "green"
            elif clade == "A":
                color = "blue"
            elif clade == "D":
                color = "red"
            else:
                color = "k"
            print(f"plotting {self.profile_meta_df.at[profile, 'ITS2 type profile']}")
            profile_ser = self.profile_count_df_rel[profile]
            dss_uids = profile_ser[profile_ser != 0].index.values
            dosoky_lat_lon_dd = defaultdict(int)
            other_lat_lon_dd = defaultdict(int)
            
            for dss in dss_uids:
                if dss in self.dosoky_dss_list:
                    dosoky_lat_lon_dd[self.dss_to_lat_lon_dict[dss]] += 1
                else:
                    other_lat_lon_dd[self.dss_to_lat_lon_dict[dss]] += 1
                all_lat_lons.add(self.dss_to_lat_lon_dict[dss])
            
            profile_to_num_dss_lat_lon[profile] = (len(dss_uids), len(other_lat_lon_dd.keys()))

            if len(dss_uids) > 0:
                if not self.dosoky_centric:
                    # We want to plot a line between all sites at which the given proifle was found
                    lat_lon_dd = {**dosoky_lat_lon_dd, **other_lat_lon_dd}
                    for lat_lon_1, lat_lon_2 in itertools.combinations(lat_lon_dd.keys(), 2):
                        # [x1: 70, x2: 90], [y1: 90, y2: 200]
                        self.ax.plot([lat_lon_1[1], lat_lon_2[1]], [lat_lon_1[0], lat_lon_2[0]], color=color, linestyle='-', linewidth=0.5, alpha=1, zorder=2)
                else:
                    # We only want to plot from each the dosoky sites to each of the other sites
                    for lat_lon, value in other_lat_lon_dd.items():
                        self.ax.plot([self.dosoky_lon, lat_lon[1]], [self.dosoky_lat, lat_lon[0]], color=color, linestyle='-', linewidth=0.5, alpha=1, zorder=2)
            else:
                foo = "bar"

        for profile, vals in profile_to_num_dss_lat_lon.items():
            print(f"{profile}: {vals[0]}, {vals[1]}")

        self.fig.savefig("harry.png", dpi=1200)
        plt.close()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        num_sample_num_site_vals_as_list = list(profile_to_num_dss_lat_lon.values())
        self.ax.scatter(x=[_[0] for _ in num_sample_num_site_vals_as_list], y=[[_[1] for _ in num_sample_num_site_vals_as_list]])
        self.ax.set_xlabel("num samples")
        self.ax.set_ylabel("num sites")
        o_xlim = self.ax.get_xlim()
        o_ylim = self.ax.get_ylim()
        self.ax.vlines(x=20, ymin=self.ax.get_ylim()[0], ymax=self.ax.get_ylim()[1], color="black")
        self.ax.hlines(y=10, xmin=self.ax.get_xlim()[0], xmax=self.ax.get_xlim()[1], color="black")
        self.ax.set_xlim(o_xlim)
        self.ax.set_ylim(o_ylim)
        self.fig.savefig("harry_scatter.png", dpi=1200)

        foo = "bar"

    def _plot_up_dosoky_centric(self):
        # Then plot this up
        # There are a considerable number of the samples that don't have a lat long associated to them
        # for these we can plot them up in the middle of the map I guess
        lats, lons, sizes = self._get_lats_lons_sizes(dd=self.lat_lon_size_dosoky_ddict)
        self.ax.scatter(lons, lats, s=sizes, c="blue")
        lats, lons, sizes = self._get_lats_lons_sizes(dd=self.lat_lon_size_other_ddict)
        self.ax.scatter(lons, lats, s=sizes, c="green")
        lats, lons, sizes = self._get_lats_lons_sizes(dd=self.lat_lon_size_no_lat_lon_ddict)
        self.ax.scatter(lons, lats, s=sizes, c="pink")

        # This works
        # Next stage is to go profile by profile in the dosoky and see where we have profiles in common.
        # We need to get a list of the profiles that are in the dosoky samples
        dosoky_samples = [k for k, v in self.sample_uid_to_dataset_uid_dict.items() if v == self.dosoky_id] 
        other_samples = [k for k, v in self.sample_uid_to_dataset_uid_dict.items() if v != self.dosoky_id]
        self.dosoky_profile_df = self.profile_count_df_rel.loc[dosoky_samples,]
        self.other_profile_df = self.profile_count_df_rel.loc[other_samples,:]
        self.dosoky_profile_df = self.dosoky_profile_df.loc[:, (self.dosoky_profile_df != 0).any(axis=0)]
        self.other_profile_df = self.other_profile_df.loc[:, (self.other_profile_df != 0).any(axis=0)]

        # Now sort by the summed relative abundance of the profiles
        self.dosoky_profile_df = self.dosoky_profile_df[self.dosoky_profile_df.sum(axis=0).sort_values(ascending=False).keys()]
        self.other_profile_df = self.other_profile_df[self.other_profile_df.sum(axis=0).sort_values(ascending=False).keys()]
        # Now work in order of these profiles
        # and pull out the number of samples it was found in the other samples
        for profile_uid in list(self.dosoky_profile_df):
            # Get the number of samples with the profile
            num_dosoky = len(self.dosoky_profile_df[profile_uid][self.dosoky_profile_df[profile_uid] != 0])
            # We will want to have the profiles counts in the other samples split up by lat lon
            # first get a set of the samples that have the profile

            # NB the profile may not be found in other samples
            try:
                other_df = self.other_profile_df[profile_uid][self.other_profile_df[profile_uid] != 0]
                print(f"{self.profile_uid_to_profile_name_dict[int(profile_uid)]} is in {num_dosoky} dosoky samples and {len(self.other_profile_df[profile_uid][self.other_profile_df[profile_uid] != 0])} other samples")
            except KeyError:
                print(f"{self.profile_uid_to_profile_name_dict[int(profile_uid)]} is in {num_dosoky} dosoky samples but not found in the other samples")
                continue
            # Now we can go dss by dss and use a default dict to count the number of samples at the given lat lon
            lat_lon_dd = defaultdict(int)
            for dss in other_df.keys():
                lat_lon_dd[self.dss_to_lat_lon_dict[dss]] += 1

            # Then we need to draw up the arrow
            # The arrow is plotted in terms of dx dy, so we need to subtract the dosoky lat and lon from the 
            # lat and lons in question
            for lat_lon, size in lat_lon_dd.items():
                dy = lat_lon[0] - self.dosoky_lat
                dx = lat_lon[1] - self.dosoky_lon
                self.ax.arrow(x=self.dosoky_lon, y=self.dosoky_lat, dx=dx, dy=dy)
                foo = "bar"
                
            foo = "bar"

    def _get_lats_lons_sizes(self, dd):
        lat_lon_as_list = list(dd.items())
        lats = [float(_[0].split(",")[0]) for _ in lat_lon_as_list]
        lons = [float(_[0].split(",")[1]) for _ in lat_lon_as_list]
        sizes = [_[1] for _ in lat_lon_as_list]
        return lats, lons, sizes

    def create_meta_study(self):
        # To generate a study we need to get all of the dataset samples from the above dataset
        data_set_sample_list = []
        for ds in self.data_sets:
            data_set_sample_list.extend(list(DataSetSample.objects.filter(data_submission_from=ds)))

        # Now we can create the new Study
        new_study = Study(name="dosoky_meta", title="dosoky_meta", location="Red Sea")
        new_study.save()
        new_study.data_set_samples.set(data_set_sample_list)
        new_study.save()
        
        print(f"{new_study.name}: {new_study.id}")

# Create a study to work with that contais the 14 datasts listed above.
dm = DosokyMeta()
# dm.create_meta_study()


# We then output that againt a DataAnalysis
# This gave us the set of outputs that are found in the directory 20210830T094047
# From here we can start to work up the data for the actual study
dm.start_map_only()
