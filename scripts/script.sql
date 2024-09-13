-- SCRIPT FOR OBTAINING DATA FOR TRAINING AND TESTING

-- !!! ALWAYS USE THE SAME GROUPDELIVERY NUMBER FOR BOTH COMMANDS !!!
-- !!! ALWAYS USE THE SAME ORDER (HERE ASCENDING) WHEN SELECTING DATA, otherwise it WON'T MATCH !!!

-- this will get the total count of all groupdelivery
-- USE THIS NUMBER TO SPLIT THE DATASET FOR TRAINING AND TESTING
select count(DISTINCT Zasilky.GroupDelivery) from Zasilky;

-- here ajdust the offset according to how many groupdelivery you want for training/testing
-- and the obtained groupdelivery number use for next command where you get the data
select DISTINCT Zasilky.GroupDelivery from Zasilky order by Zasilky.GroupDelivery ASC LIMIT 1 OFFSET 120552;

-- this command get all INPUT DATA until the groupdelivery number 
select Z.GroupDelivery, Z.Product, K.X, K.Y, K.Z, K.Weight, Z.Qty from Zasilky Z left join KD K
    on
        Z.Product = K.Product and K.ContentQty = 1 -- every item is joined with its parameters for single item, (package's sizes and data are not considered, everything is considered as single item)
    where Z.GroupDelivery <= 12012131 order by GroupDelivery ASC
    ;

-- this command get all OUTPUT DATA (meaning data for training for computing loss function) until the groupdelivery number 
select distinct Z.Expedition, Z.UsedCarton, Z.GroupDelivery from Zasilky Z
    where Z.GroupDelivery <= 12012131 order by GroupDelivery ASC-- ASC/DESC must be identical to command before
    ;
