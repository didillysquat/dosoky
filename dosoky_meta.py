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

class DosokyMeta:
    def __init__(self):
        if os.path.exists("cache/meta_df.p"):
            self.meta_df = pickle.load(open("cache/meta_df.p", "rb"))
        else:
            self.meta_df = pd.read_csv("20210830T094047/post_med_seqs/161_20210712_DBV_20210830T094047.seqs.absolute.meta_only.txt", sep="\t")
            self.meta_df.set_index("sample_uid", inplace=True, drop=True)
            pickle.dump(self.meta_df, open("cache/meta_df.p", "wb"))
        
        if os.path.exists("cache/post_med_count_df_abs.p"):
            self.post_med_count_df_abs = pickle.load(open("cache/post_med_count_df_abs.p", "rb"))
        else:
            self.post_med_count_df_abs = pd.read_csv("20210830T094047/post_med_seqs/161_20210712_DBV_20210830T094047.seqs.absolute.abund_only.txt", sep="\t")
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
            self.profile_meta_df = pd.read_csv("20210830T094047/its2_type_profiles/161_20210712_DBV_20210830T094047.profiles.meta_only.txt", sep="\t")
            self.profile_meta_df.set_index("ITS2 type profile UID", inplace=True, drop=True)
            pickle.dump(self.profile_meta_df, open("cache/profile_meta_df.p", "wb"))
        self.profile_uid_to_profile_name_dict = {k:v for k, v in self.profile_meta_df["ITS2 type profile"].items()}

        if os.path.exists("cache/profile_count_df_rel.p"):
            self.profile_count_df_rel = pickle.load(open("cache/profile_count_df_rel.p", "rb"))
        else:
            self.profile_count_df_rel = pd.read_csv("20210830T094047/its2_type_profiles/161_20210712_DBV_20210830T094047.profiles.relative.abund_only.txt", sep="\t")
            self.profile_count_df_rel.set_index("sample_uid", inplace=True, drop=True)
            pickle.dump(self.profile_count_df_rel, open("cache/profile_count_df_rel.p", "wb"))
        
        self.names_of_data_sets_to_use = [
            "hume_et_al_2020", "20201018_schupp_all", "20190617_ziegler",
            "120418_buitrago_sp_v0.3.8", "20190612_dosoky", "240418_buitrago_sp_v0.3.8",
            "20210506_terraneo", "20210412_ziegler", "20171009_roberto_sp_v0.3.8", "terraneo_et_al_2019",
            "20170724_smith_singapore_v0_3_14",  "20190612_cardenas", "20190612_rossbach", "gardener_et_al_2019"
        ]

        self.data_sets = [DataSet.objects.get(name=_) for _ in self.names_of_data_sets_to_use]

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
                self.sample_uid_to_dataset_uid_dict[dss.id] = ds.id
                self.sample_uid_to_dataset_name_dict[dss.id] = ds.name
        
        self.meta_df["data_set_name"] = [self.sample_uid_to_dataset_name_dict[_] for _ in self.meta_df.index]
        self.meta_df["data_set_uid"] = [self.sample_uid_to_dataset_uid_dict[_] for _ in self.meta_df.index]
        foo = "bar"

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

        # Make base plot of the sites
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        # We want to plot up a scatter of the lat and longs
        # with size
        # I think it is helpful to have a seperate set of points for the dosoky, no lat long, and others
        lat_lon_size_dosoky_ddict = defaultdict(int)
        lat_lon_size_no_lat_lon_ddict = defaultdict(int)
        lat_lon_size_other_ddict = defaultdict(int)
        self.dss_to_lat_lon_dict = {}
        sample_uid_no_lat_lon = []
        no_lat_dd = defaultdict(int)
        for dss_uid, ser in self.meta_df.iterrows():
            if "999" in str(ser["collection_latitude"]):
                no_lat_dd[self.sample_uid_to_dataset_name_dict[dss_uid]] += 1
                print(f"No lat lon for {ser['sample_name']}")
                sample_uid_no_lat_lon.append(dss_uid)
                # Assiciate a lat long that puts them in the top right of the map for the time being
                lat_lon_size_no_lat_lon_ddict[",".join(["22", "90"])] += 1
                self.dss_to_lat_lon_dict[dss_uid] = (22,90)
            else:
                if self.sample_uid_to_dataset_uid_dict[dss_uid] == self.dosoky_id:
                    lat_lon_size_dosoky_ddict[",".join([str(ser["collection_latitude"]), str(ser["collection_longitude"])])] += 1
                else:
                    lat_lon_size_other_ddict[",".join([str(ser["collection_latitude"]), str(ser["collection_longitude"])])] += 1
                self.dss_to_lat_lon_dict[dss_uid] = (ser["collection_latitude"], ser["collection_longitude"])
        # Get the data set names of samples with no lat lon
        no_lat_lon_ds_set = {self.data_set_uid_to_data_set_name_dict[self.sample_uid_to_dataset_uid_dict[_]] for _ in sample_uid_no_lat_lon}
        
        # Then plot this up
        # There are a considerable number of the samples that don't have a lat long associated to them
        # for these we can plot them up in the middle of the map I guess
        lats, lons, sizes = self._get_lats_lons_sizes(dd=lat_lon_size_dosoky_ddict)
        self.ax.scatter(lons, lats, s=sizes, c="blue")
        lats, lons, sizes = self._get_lats_lons_sizes(dd=lat_lon_size_other_ddict)
        self.ax.scatter(lons, lats, s=sizes, c="green")
        lats, lons, sizes = self._get_lats_lons_sizes(dd=lat_lon_size_no_lat_lon_ddict)
        self.ax.scatter(lons, lats, s=sizes, c="pink")

        # This works
        # Next stage is to go profile by profile in the dosoky and see where we have profiles in common.
        # We need to get a list of the profiles that are in the dosoky samples
        dosoky_samples = [k for k, v in self.sample_uid_to_dataset_uid_dict.items() if v == self.dosoky_id] 
        other_samples = [k for k, v in self.sample_uid_to_dataset_uid_dict.items() if v != self.dosoky_id]
        self.dosoky_profile_df = self.profile_count_df_rel.loc[dosoky_samples,]
        self.other_profile_df = self.profile_count_df_rel.loc[other_samples,]
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

            # NB the common may not be found in other samples
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
            foo = "bar"

        foo = "bar"
        self.fig.savefig("harry.png")

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
