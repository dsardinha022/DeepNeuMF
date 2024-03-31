# Davina Sardinha
# Copyreight (c) Desco Industries. All rights reserved.
# Licensed under the MIT License.

DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_ITEMDESC_COL = "itemDesc"
DEFAULT_RATING_COL = "rating"
DEFAULT_BRAND_COL = "brand"
DEFAULT_POSITIVE_COL = "positive_samples"
DEFAULT_NEGATIVE_COL = "negative_samples"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_PREDICTION_COL = "prediction"

COLUMN_DICT = {
    "column_user": DEFAULT_USER_COL,
    "column_item": DEFAULT_ITEM_COL,
    "column_itemdesc": DEFAULT_ITEMDESC_COL,
    "column_rating": DEFAULT_RATING_COL,
    "column_brand": DEFAULT_BRAND_COL,
    "column_time": DEFAULT_TIMESTAMP_COL,
    "column_prediction": DEFAULT_PREDICTION_COL,
}

DEFAULT_K = 15
DEFAULT_THRESHOLD = 10
DEFAULT_SEED = 37