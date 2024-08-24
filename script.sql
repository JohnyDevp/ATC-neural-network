select Expedition, ExpUnit, UsedCarton from Zasilky limit 10;

select Z.GroupDelivery, Z.Product, Z.ExpUnit, Z.Qty, K.Product, K.Unit, K.Weight, K.ContentQty from Zasilky as Z right join KD K on
    Z.Product = K.Product
        where K.ContentQty = (SELECT ContentQty from KD where Z.Product = KD.Product order by ABS(KD.ContentQty - Z.Qty) ASC LIMIT 1)
        limit 1;

-- command to get input data
-- here is obtained the total number of all group delivery
-- this is used to split the dataset for training and test ... we take 20% for testing
-- edit appropriate number meaning the edge in the dataset in form of group delivery number
select count(DISTINCT Zasilky.GroupDelivery) from Zasilky;

select DISTINCT Zasilky.GroupDelivery from Zasilky order by Zasilky.GroupDelivery ASC LIMIT 1 OFFSET 120552;

select Z.GroupDelivery, Z.Product, K.X, K.Y, K.Z, K.Weight, Z.Qty from Zasilky Z left join KD K
    on
        Z.Product = K.Product and K.ContentQty = 1 -- every item is joined with its parameters for single item, (package's sizes and data are not considered, everything is considered as single item)
    where Z.GroupDelivery <= 12012131 order by GroupDelivery ASC
    ;

select distinct Z.Expedition, Z.UsedCarton, Z.GroupDelivery from Zasilky Z
    where Z.GroupDelivery <= 12012131 order by GroupDelivery ASC-- this must be identical to command before
    ;

select DISTINCT UsedCarton from Zasilky where GroupDelivery=12012131;


