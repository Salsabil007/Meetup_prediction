import pandas as pd
import numpy as np

poi = pd.read_csv("user_poi.csv")
data = poi

data = data.drop(data.columns[[0,3,4,5,6,7,8]],1)
data = data.drop_duplicates()

day_act = []
night_act = []
meet_imp = []
for ind in data.index:
    temp = poi[poi['venueid'] == data['venueid'][ind]]
    d = temp[temp['hour'] >= 7]
    d = temp[temp['hour'] <= 19]
    n = temp[temp['hour'] < 7]
    n1 = temp[temp['hour'] > 19]
    n = n.append(n1, ignore_index = True)
    day_act.append(len(d)/len(temp))
    night_act.append(len(n)/len(temp))
    
data['day_act'] = day_act
data['night_act'] = night_act
data.to_csv("poi_feature_intermideate.csv", index = False)
venue1 = ['Bar', 'Pub', 'BBQ Joint', 'Restaurant', 'American Restaurant' ,'Pizza Place','Dive Bar', 'Sports Bar', 'Wine Bar', 'Convention Center', 'Ice Cream Shop','Hot Dog Joint' ,'Dim Sum Restaurant', 'Comedy Club',
 'Casino' ,'Rock Club','Diner', 'Hotel' ,'Chinese Restaurant','Burger Joint', 'Coffee Shop','Seafood Restaurant','Steakhouse' ,'Fast Food Restaurant','Dessert Shop','Bed & Breakfast','Internet Cafe','Subway',
 'Breakfast Spot' ,'Cocktail Bar' ,'Speakeasy' ,'Brewery','New American Restaurant', 'Asian Restaurant','Juice Bar', 'Pool Hall','Ski Lodge', 'Mongolian Restaurant' , 'Beer Garden','Mexican Restaurant', 'German Restaurant', 'Whisky Bar','Tea Room',
 'Eastern European Restaurant', 'Cajun / Creole Restaurant', 'Bakery','Australian Restaurant','Brazilian Restaurant',
  'Historic Site' ,'Korean Restaurant','Donut Shop', 'Peruvian Restaurant','Snack Place',
 'Karaoke Bar' , 'Gay Bar', 'Italian Restaurant','Cheese Shop','Middle Eastern Restaurant',
 'Mac & Cheese Joint','Indonesian Restaurant','Sake Bar', 'Candy Store' ,'Fish & Chips Shop','Burrito Place',
  'Salad Place' ,'Gastropub', 'Sandwich Place','Falafel Restaurant','Spanish Restaurant' ,'Sushi Restaurant', 'Jazz Club',
 'Deli / Bodega' ,'Plaza',  'Mall', 'Taco Place','Winery', 'Thai Restaurant', 'Japanese Restaurant',
 'Piano Bar' , 'Caribbean Restaurant', 'Cuban Restaurant', 'Arepa Restaurant' ,'Filipino Restaurant', 'Cafeteria', 'Hookah Bar',
  'Vegetarian / Vegan Restaurant','Food Truck' ,'Nightclub', 'Indian Restaurant','Ramen /  Noodle House','Yogurt', 'African Restaurant',
  'French Restaurant','Swiss Restaurant','Vietnamese Restaurant','Scandinavian Restaurant',
 'Mediterranean Restaurant','Food', 'Gourmet Shop', 'Latin American Restaurant','Hotel Bar' ,'Fair', 'Greek Restaurant', 'Wings Joint','South American Restaurant','Cupcake Shop','Tapas Restaurant','Food Court','Food & Drink Shop',
 'Turkish Restaurant', 'Shrine', 'Gluten-free Restaurant','Fried Chicken Joint','Bagel Shop',
 'Southern / Soul Food Restaurant','Other Nightlife']

venue2 =['General Entertainment','Baseball Stadium', 'Park', 'Music Venue', 'Other Great Outdoors', 'Pool', 'Event Space','Science Museum','Beach','Coworking Space','College Auditorium',
 'History Museum', 'Strip Club' ,'Movie Theater' ,'Theme Park', 'Other Event','College Football Field','Art Gallery','Resort','Hockey Arena', 'Theater' ,'Moving Target', 'Museum', 'Gym / Fitness Center',
 'Field','Library' ,'Arcade' ,'Playground', 'Bowling Alley', 'Multiplex','Bridge' ,'Pier', 'Sculpture Garden' ,'Garden',
'Performing Arts Venue','Church','Soccer Stadium' ,'Lounge', 'Concert Hall', 'Soccer Field', 'Motel',
 'Gym', 'Bookstore','Racetrack', 'Basketball Stadium','Ski Area','Vineyard',
 'Opera House', 'Stadium','Tennis','Garden Center', 
 'College Gym', 'Hot Spring', 'Arts & Entertainment' ,'Dance Studio',
 'Housing Development','Baseball Field','Harbor / Marina', 'Light Rail', 'Skating Rink','Football Stadium' ,'College Hockey Rink', 'Golf Course',
  'Gaming Cafe','Tennis Court' ,'College Baseball Diamond', 'College Soccer Field','College Basketball Court',
 'Skate Park', 'Boarding House','Aquarium', 'Hiking Trail' ,'Public Art','Athletic & Sport',
 'Outdoors & Recreation','Dog Run','Photography Lab' ,'Yoga Studio', 'Nightlife Spot', 'College Theater' ,'Rest Area', 'Music Store' ,'Lake',
 'Theme Park Ride / Attraction','Water Park', 'Design Studio','Zoo', 'Athletics & Sports','Martial Arts Dojo',
 'College Rec Center','Lighthouse','College Stadium', 'Video Store' ,'Indie Theater', 'Travel Lounge',
 'Art Museum','Indie Movie Theater', 'Scenic Lookout','Monument / Landmark']

venue3 = ['Tech Startup','Office''University','Government Building','College Lab','High School','General College & University','College Classroom',
 'College Administrative Building', 'Student Center', 'Medical School','School','Fraternity House', 'Factory', 'College Arts Building', 
 'Law School', 'City' ,'Elementary School','College Technology Building',
    'College Academic Building', 'College Quad', 
    'College Library','College Engineering Building','College Cafeteria',
     'College Science Building','Conference Room']

venue4 = ['Gift Shop','Farmers Market','Toy / Game Store' ,'Automotive Shop','Grocery Store','Department Store','Liquor Store','Flea Market','Pet Store', 'Board Shop' ,
'Jewelry Store','Boutique', 'Record Shop','Clothing Store','Arts & Crafts Store',
 'Thrift / Vintage Store','Miscellaneous Shop','Bike Shop','Market','Cosmetics Shop', 'Medical Center',
  'Paper / Office Supplies Store','Smoke Shop',"Women's Store",  'Tourist Information Center' ,'Spa / Massage',
 'Travel & Transport' , 'Hobby Shop','Track' ,'Car Dealership' , 'Flower Shop' ,'Shoe Store',
 'Rental Car Location','Convenience Store', 'Antique Shop', 'Tanning Salon' ,'Veterinarian', 
 'Video Game Store', 'Bridal Shop','Wine Shop' ,'Accessories Store',
 'Mobile Phone Shop', 'Kids Store',"Men's Store",'College Bookstore','Tattoo Parlor','Furniture / Home Store']

for ind in data.index:
    if data['venue catagory'][ind] in venue1:
        meet_imp.append(500)
    elif data['venue catagory'][ind] in venue2:
        meet_imp.append(350)
    elif data['venue catagory'][ind] in venue3:
        meet_imp.append(250)
    elif data['venue catagory'][ind] in venue4:
        meet_imp.append(100)
    else:
        meet_imp.append(50)
data['meet_imp'] = meet_imp
data.to_csv("poi_feature.csv", index = False)