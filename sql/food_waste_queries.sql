CREATE DATABASE Food_Waste_DB;
USE Food_Waste_DB;
show tables;
SELECT * 
FROM food_waste_feature_engineered;

SELECT COUNT(*) AS Total_Records
FROM food_waste_feature_engineered;

DESCRIBE food_waste_feature_engineered;
SELECT ROUND(SUM(Waste_Quantity), 2) AS Total_Waste_Quantity
FROM food_waste_feature_engineered;
SELECT ROUND(SUM(Waste_Cost), 2) AS Total_Waste_Cost
FROM food_waste_feature_engineered;

SELECT ROUND(AVG(Waste_Quantity), 2) AS Average_Waste
FROM food_waste_feature_engineered;

SELECT
    Category,
    ROUND(SUM(Waste_Quantity), 2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY Category
ORDER BY Total_Waste DESC;

SELECT
    Store_Name,
    ROUND(SUM(Waste_Quantity), 2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY Store_Name
ORDER BY Total_Waste DESC
LIMIT 10;

SELECT
    City,
    ROUND(SUM(Waste_Quantity), 2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY City
ORDER BY Total_Waste DESC;

SELECT
    State,
    ROUND(SUM(Waste_Quantity), 2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY State
ORDER BY Total_Waste DESC;

SELECT
    Region,
    ROUND(SUM(Waste_Quantity), 2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY Region
ORDER BY Total_Waste DESC;

SELECT
    Demand_Level,
    COUNT(*) AS Total_Products,
    ROUND(AVG(Waste_Quantity),2) AS Average_Waste,
    ROUND(SUM(Waste_Quantity),2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY Demand_Level
ORDER BY Total_Waste DESC;

SELECT
    Weather,
    COUNT(*) AS Total_Records,
    ROUND(AVG(Waste_Quantity),2) AS Average_Waste,
    ROUND(SUM(Waste_Quantity),2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY Weather
ORDER BY Total_Waste DESC;

SELECT
    Promotion,
    ROUND(AVG(Waste_Quantity),2) AS Average_Waste,
    ROUND(SUM(Waste_Quantity),2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY Promotion;

SELECT
    Weekend,
    ROUND(AVG(Waste_Quantity),2) AS Average_Waste,
    ROUND(SUM(Waste_Quantity),2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY Weekend;

SELECT
    Holiday,
    ROUND(AVG(Waste_Quantity),2) AS Average_Waste,
    ROUND(SUM(Waste_Quantity),2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY Holiday;

SELECT
    Month,
    ROUND(SUM(Waste_Quantity),2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY Month
ORDER BY Month;

SELECT
    Product_Name,
    ROUND(SUM(Waste_Quantity),2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY Product_Name
ORDER BY Total_Waste DESC
LIMIT 10;

SELECT
    Product_Name,
    ROUND(SUM(Waste_Cost),2) AS Total_Waste_Cost
FROM food_waste_feature_engineered
GROUP BY Product_Name
ORDER BY Total_Waste_Cost DESC
LIMIT 10;

SELECT
    Store_Name,
    ROUND(SUM(Waste_Quantity),2) AS Total_Waste
FROM food_waste_feature_engineered
GROUP BY Store_Name
HAVING SUM(Waste_Quantity) > 500
ORDER BY Total_Waste DESC;

SELECT
    Store_Type,
    ROUND(AVG(Waste_Quantity),2) AS Average_Waste
FROM food_waste_feature_engineered
GROUP BY Store_Type
ORDER BY Average_Waste DESC;

SELECT
    Product_Name,
    Waste_Quantity,
    CASE
        WHEN Waste_Quantity < 5 THEN 'Low Waste'
        WHEN Waste_Quantity BETWEEN 5 AND 15 THEN 'Medium Waste'
        ELSE 'High Waste'
    END AS Waste_Category
FROM food_waste_feature_engineered;

SELECT
    Product_Name,
    Inventory_Remaining,
    CASE
        WHEN Inventory_Remaining < 20 THEN 'Low Stock'
        WHEN Inventory_Remaining BETWEEN 20 AND 100 THEN 'Normal Stock'
        ELSE 'High Stock'
    END AS Inventory_Status
FROM food_waste_feature_engineered;

SELECT
    Product_Name,
    Waste_Quantity
FROM food_waste_feature_engineered
WHERE Waste_Quantity = (
    SELECT MAX(Waste_Quantity)
    FROM food_waste_feature_engineered
);

SELECT
    Product_Name,
    Waste_Quantity
FROM food_waste_feature_engineered
WHERE Waste_Quantity >
(
    SELECT AVG(Waste_Quantity)
    FROM food_waste_feature_engineered
)
ORDER BY Waste_Quantity DESC;

SELECT
    Store_Name,
    SUM(Waste_Quantity) AS Total_Waste,
    RANK() OVER(
        ORDER BY SUM(Waste_Quantity) DESC
    ) AS Store_Rank
FROM food_waste_feature_engineered
GROUP BY Store_Name;

SELECT
    Category,
    SUM(Waste_Quantity) AS Total_Waste,
    DENSE_RANK() OVER(
        ORDER BY SUM(Waste_Quantity) DESC
    ) AS Category_Rank
FROM food_waste_feature_engineered
GROUP BY Category;

WITH ProductWaste AS
(
    SELECT
        Category,
        Product_Name,
        SUM(Waste_Quantity) AS Total_Waste,
        RANK() OVER(
            PARTITION BY Category
            ORDER BY SUM(Waste_Quantity) DESC
        ) AS rnk
    FROM food_waste_feature_engineered
    GROUP BY Category, Product_Name
)

SELECT *
FROM ProductWaste
WHERE rnk = 1;

SELECT
    Brand,
    ROUND(AVG(Waste_Quantity),2) AS Average_Waste
FROM food_waste_feature_engineered
GROUP BY Brand
ORDER BY Average_Waste DESC;

SELECT
    Storage_Type,
    ROUND(AVG(Waste_Quantity),2) AS Average_Waste
FROM food_waste_feature_engineered
GROUP BY Storage_Type
ORDER BY Average_Waste DESC;

SELECT
    Waste_Risk,
    COUNT(*) AS Total_Products
FROM food_waste_feature_engineered
GROUP BY Waste_Risk;