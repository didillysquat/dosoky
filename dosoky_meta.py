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
import os
import pickle
import itertools

class DosokyMeta:
    def __init__(self, plot_no_lat_lon=False):
        if os.path.exists("cache/meta_df.p"):
            self.meta_df = pickle.load(open("cache/meta_df.p", "rb"))
        else:
            self.meta_df = pd.read_csv("20210902T051904/post_med_seqs/169_20210830_DBV_20210902T051904.seqs.absolute.meta_only.txt", sep="\t")
            self.meta_df.set_index("sample_uid", inplace=True, drop=True)
            pickle.dump(self.meta_df, open("cache/meta_df.p", "wb"))
        
        if os.path.exists("cache/post_med_count_df_abs.p"):
            self.post_med_count_df_abs = pickle.load(open("cache/post_med_count_df_abs.p", "rb"))
        else:
            self.post_med_count_df_abs = pd.read_csv("20210902T051904/post_med_seqs/169_20210830_DBV_20210902T051904.seqs.absolute.abund_only.txt", sep="\t")
            self.post_med_count_df_abs.set_index("sample_uid", inplace=True, drop=True)
            pickle.dump(self.post_med_count_df_abs, open("cache/post_med_count_df_abs.p", "wb"))
        
        if os.path.exists("cache/post_med_count_df_rel.p"):
            self.post_med_count_df_rel = pickle.load(open("cache/post_med_count_df_rel.p", "rb"))
        else:
            self.post_med_count_df_rel = self.post_med_count_df_abs.div(self.post_med_count_df_abs.sum(axis=1), axis=0)
            pickle.dump(self.post_med_count_df_rel, open("cache/post_med_count_df_rel.p", "wb"))
        
        if os.path.exists("cache/profile_meta_df.p"):
            self.profile_meta_df = pickle.load(open("cache/profile_meta_df.p", "rb"))
        else:
            self.profile_meta_df = pd.read_csv("20210902T051904/its2_type_profiles/169_20210830_DBV_20210902T051904.profiles.meta_only.txt", sep="\t")
            self.profile_meta_df.set_index("ITS2 type profile UID", inplace=True, drop=True)
            pickle.dump(self.profile_meta_df, open("cache/profile_meta_df.p", "wb"))
        self.profile_uid_to_profile_name_dict = {k:v for k, v in self.profile_meta_df["ITS2 type profile"].items()}

        if os.path.exists("cache/profile_count_df_rel.p"):
            self.profile_count_df_rel = pickle.load(open("cache/profile_count_df_rel.p", "rb"))
        else:
            self.profile_count_df_rel = pd.read_csv("20210902T051904/its2_type_profiles/169_20210830_DBV_20210902T051904.profiles.relative.abund_only.txt", sep="\t")
            self.profile_count_df_rel.set_index("sample_uid", inplace=True, drop=True)
            # make the headers int
            self.profile_count_df_rel.columns = [int(_) for _ in list(self.profile_count_df_rel)]
            pickle.dump(self.profile_count_df_rel, open("cache/profile_count_df_rel.p", "wb"))
        
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
        self.profile_count_df_rel = self.profile_count_df_rel.loc[index_keep, header_keep]
        # finally remove the samples from the self.meta_df
        self.meta_df = self.meta_df.loc[index_keep,]

        # Optional, remove those samples that don't have a lat lon associated to them
        if not plot_no_lat_lon:
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

        self.names_of_data_sets_to_use = [
            "hume_et_al_2020", "20201018_schupp_all", "20190617_ziegler",
            "120418_buitrago_sp_v0.3.8", "20190612_dosoky", "240418_buitrago_sp_v0.3.8",
            "20210506_terraneo", "20210412_ziegler", "20171009_roberto_sp_v0.3.8", "terraneo_et_al_2019",
            "20170724_smith_singapore_v0_3_14",  "20190612_cardenas", "20190612_rossbach", "gardener_et_al_2019",
            "20170506_roberto_sp_v0.3.8", "20171017_roberto_sp_v0.3.8", "20171114_roberto_sp_v0.3.8"
        ]

        self.data_sets = [DataSet.objects.get(name=_) for _ in self.names_of_data_sets_to_use]

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

        # Now check to see if there were any profiles that were only found in the schupp samples that we removed
        # that now have no counts in them
        profiles_only_in_schupp = self.profile_count_df_rel.any(axis=0)
        print(f"{sum(profiles_only_in_schupp)} profiles were only found in the schupp larval samples")
        self.profile_count_df_rel = self.profile_count_df_rel.loc[:,self.profile_count_df_rel.any(axis=0)]
        print(f"{len(self.profile_count_df_rel.columns)} profiles left after removing")

        # Sort the profile count df by abundance of the profiles
        self.profile_count_df_rel = self.profile_count_df_rel[self.profile_count_df_rel.sum(axis=0).sort_values(ascending=False).index.values]

        # make a dict that is sample_uid to dataset uid
        # Then append this information to the self.meta_df
        self.sample_uid_to_dataset_uid_dict = {}
        self.sample_uid_to_dataset_name_dict = {}
        self.data_set_uid_to_data_set_name_dict = {}
        for ds in self.data_sets:
            if ds.name == "20190612_dosoky":
                self.dosoky_id = ds.id
            self.data_set_uid_to_data_set_name_dict[ds.id] = ds.name
            for dss in DataSetSample.objects.filter(data_submission_from=ds):
                # If it is one of the datasetsamples that we kicked out due to being with a dodgy
                # don't include it.
                if dss.id not in self.meta_df.index:
                    continue
                self.sample_uid_to_dataset_uid_dict[dss.id] = ds.id
                self.sample_uid_to_dataset_name_dict[dss.id] = ds.name
        
        self.meta_df["data_set_name"] = [self.sample_uid_to_dataset_name_dict[_] for _ in self.meta_df.index]
        self.meta_df["data_set_uid"] = [self.sample_uid_to_dataset_uid_dict[_] for _ in self.meta_df.index]
        self.dosoky_lat, self.dosoky_lon = 25.54, 34.64
        foo = "bar"

        # Make base plot of the sites
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        self.ax.set_xlabel("longitude Decimal Degree")
        self.ax.set_ylabel("latitude Decimal Degree")
        # We want to plot up a scatter of the lat and longs
        # with size
        # I think it is helpful to have a seperate set of points for the dosoky, no lat long, and others
        self.lat_lon_size_dosoky_ddict = defaultdict(int)
        self.lat_lon_size_no_lat_lon_ddict = defaultdict(int)
        self.lat_lon_size_other_ddict = defaultdict(int)
        self.dss_to_lat_lon_dict = {}
        self.sample_uid_no_lat_lon = []
        no_lat_dd = defaultdict(int)
        for dss_uid, ser in self.meta_df.iterrows():
            if "999" in str(ser["collection_latitude"]):
                no_lat_dd[self.sample_uid_to_dataset_name_dict[dss_uid]] += 1
                print(f"No lat lon for {ser['sample_name']}")
                self.sample_uid_no_lat_lon.append(dss_uid)
                # Assiciate a lat long that puts them in the top right of the map for the time being
                self.lat_lon_size_no_lat_lon_ddict[",".join(["22", "90"])] += 1
                self.dss_to_lat_lon_dict[dss_uid] = (22,90)
            else:
                if self.sample_uid_to_dataset_uid_dict[dss_uid] == self.dosoky_id:
                    self.lat_lon_size_dosoky_ddict[",".join([str(ser["collection_latitude"]), str(ser["collection_longitude"])])] += 1
                else:
                    self.lat_lon_size_other_ddict[",".join([str(ser["collection_latitude"]), str(ser["collection_longitude"])])] += 1
                self.dss_to_lat_lon_dict[dss_uid] = (ser["collection_latitude"], ser["collection_longitude"])
        # Get the data set names of samples with no lat lon
        no_lat_lon_ds_set = {self.data_set_uid_to_data_set_name_dict[self.sample_uid_to_dataset_uid_dict[_]] for _ in self.sample_uid_no_lat_lon}

    def start(self):
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

        foo = "bar"
        # At this point we have the connectivity on a dosoky centric way plotted.
        # The other approach that will be interesting will be to plot the interconnectivity between all profiles
        # The way to do this will be to go profile by profile
        # For each profile get the samples that have that profiles
        # Then make a count dict of the lat lons associated with those
        # Then we want to find all pairwise comparisions between those and plot a line to represent these.
        # That will be our network.
        for profile in self.profile_count_df_rel:
            print(f"plotting {self.profile_meta_df.at[profile, 'ITS2 type profile']}")
            profile_ser = self.profile_count_df_rel[profile]
            dss_uids = profile_ser[profile_ser != 0].index.values
            lat_lon_dd = defaultdict(int)
            for dss in dss_uids:
                lat_lon_dd[self.dss_to_lat_lon_dict[dss]] += 1
            for lat_lon_1, lat_lon_2 in itertools.combinations(lat_lon_dd.keys(), 2):
                self.ax.plot([lat_lon_1[1], lat_lon_2[1]], [lat_lon_1[0], lat_lon_2[0]], color='k', linestyle='-', linewidth=0.5, alpha=0.2)

        self.fig.savefig("harry.png", dpi=1200)

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
dm.start()
