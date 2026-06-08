import numpy as np
import pandas as pd


accomodation = [
    'apartments', 
    'barracks',
    'bungalow',
    'cabin',
    'detached',
    'annexe',
    'dormitory',
    'farm',	
    'ger',
    'hotel',
    'house',
    'houseboat',
    'residential',
    'semidetached_house',
    'static_caravan',
    'stilt_house',
    'terrace',
    'tree_house',
    'trullo'
]

commercial = [
    'commercial',
    'industrial',
    'kiosk',
    'office',
    'retail',
    'supermarket',
    'warehouse'
]

religious = [
    'religious',
    'cathedral',
    'chapel',
    'church',
    'kingdom_hall',
    'monastery',
    'mosque',
    'presbytery',
    'shrine',
    'synagogue',
    'temple'
]

civic = [
    'bakehouse',
    'bridge',
    'civic',
    'clock_tower',
    'college',
    'fire_station',
    'government',
    'gatehouse',
    'hospital',
    'kindergarten',
    'museum',
    'public',
    'school',
    'toilets',
    'train_station',
    'transportation',
    'university'
]

agricultural = [
    'barn',
    'conservatory',
    'cowshed',
    'farm_auxiliary',
    'greenhouse',
    'slurry_tank',
    'stable',
    'sty',
    'livestock'
]

sports = [
    'grandstand',
    'pavilion',
    'riding_hall',
    'sports_hall',
    'sports_centre',
    'stadium'
]


storage = [
    'allotment_house',
    'boathouse',
    'hangar',
    'hut',
    'shed'
]


cars = [
    'carport',
    'garage',
    'garages',
    'parking',
]


technical = [
    'digester',
    'service',
    'tech_cab',
    'transformer_tower',
    'water_tower',
    'storage_tank',
    'silo'
]


other = [
    'beach_hut',
    'bunker',
    'castle',
    'construction',
    'container',
    'guardhouse',
    'military',
    'outbuilding',
    'pagoda',
    'quonset_hut',
    'roof',
    'ruins',
    'ship',
    'tent',
    'tower',
    'triumphal_arch',
    'windmill'
]


def classify_type(word):
    if word in accomodation:
        return 'Жилые'
    elif word in commercial:
        return 'Коммерческие'
    elif word in religious:
        return 'Религиозные'
    elif word in civic:
        return 'Общественные'
    elif word in agricultural:
        return 'Сельскохозяйственные'
    elif word in sports:
        return 'Спортивные'
    elif word in storage:
        return 'Складские'
    elif word in cars:
        return 'Автомобильные'
    elif word in technical:
        return 'Технические'
    elif word in other:
        return 'Прочие'
    else:
        return 'Пользовательские типы'


def building_amount(threshold: int = 1000, multiplier: int = 2000, correction: float = 1):
    building_count = pd.read_csv('combined_building_count.csv')

    building_count['category'] = building_count['type'].apply(classify_type)
    
    building_count = building_count.iloc[1:]
    building_count = building_count[(building_count['building_count'] > threshold) & (building_count['category'] != 'Пользовательские типы')]
    
    building_count['z_score'] = (building_count['building_count'] - building_count['building_count'].mean()) / building_count['building_count'].std()
    building_count['z_score_sigmoid'] = 1 / (1 + correction * np.exp(-building_count['z_score']))
    
    building_count['multiplied'] = (building_count['z_score_sigmoid'] * multiplier).astype(int)
    amount_list = building_count[['type', 'multiplied']].values.tolist()
    
    # return building_count
    # return building_count["multiplied"].sum()
    return amount_list

def building_amount_zipf(threshold: int = 1000, multiplier: int = 2000, correction: float = 1):
    building_count = pd.read_csv('combined_building_count.csv')

    building_count['category'] = building_count['type'].apply(classify_type)
    
    building_count = building_count.iloc[1:]
    building_count = building_count[(building_count['building_count'] > threshold) & (building_count['category'] != 'Пользовательские типы')]
    
    building_count['z_score'] = (building_count['building_count'] - building_count['building_count'].mean()) / building_count['building_count'].std()
    building_count['rank'] = building_count.index
    building_count['multiplied'] = (building_count['building_count'] * np.log(building_count['rank']+1) / correction).astype(int)
    
    amount_list = building_count[['type', 'multiplied']].values.tolist()

    return amount_list
    # return building_count
    # return building_count["multiplied"].sum()

if __name__ == "__main__":
    print(building_amount_zipf(threshold = 1000, correction = 5))


