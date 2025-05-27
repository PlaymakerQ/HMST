from utils.tools import MyTool
name_dict = {
    "user_id": 'user_id',
    "poi_id": 'POI_id',
    "cat_id": 'POI_catid',
    "catid_code": 'POI_catid_code',
    "cat_name": 'POI_catname',
    "latitude": 'latitude',
    "longitude": 'longitude',
    "timezone": 'timezone',
    "utc_time": 'UTC_time',
    "local_time": 'local_time',
    "day_in_week": 'day_of_week',
    "norm_in_day_time": 'norm_in_day_time',
    "trajectory_id": 'trajectory_id',
    "norm_day_shift": 'norm_day_shift',
    "norm_relative_time": 'norm_relative_time',
}


col_name = MyTool.convert_dict_to_object(name_dict)