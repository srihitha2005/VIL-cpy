
-- Users
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100),
    password_hash TEXT NOT NULL,
    role ENUM('admin', 'manager') DEFAULT 'manager'
);

-- Restaurants
CREATE TABLE restaurants (
    restaurant_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    location VARCHAR(255),
    manager_id INT,
    FOREIGN KEY (manager_id) REFERENCES users(user_id)
);

-- Menu Items
CREATE TABLE menus (
    menu_id INT PRIMARY KEY AUTO_INCREMENT,
    restaurant_id INT,
    item_name VARCHAR(100),
    price DECIMAL(10, 2),
    category VARCHAR(50),
    available BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
);

-- Orders
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    restaurant_id INT,
    order_date DATETIME,
    total_amount DECIMAL(10, 2),
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
);

-- Order Items
CREATE TABLE order_items (
    item_id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT,
    menu_id INT,
    quantity INT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (menu_id) REFERENCES menus(menu_id)
);

-- Inventory
CREATE TABLE inventory (
    inventory_id INT PRIMARY KEY AUTO_INCREMENT,
    restaurant_id INT,
    item_name VARCHAR(100),
    quantity INT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
);

-- ML Models (Optional - for tracking model versions)
CREATE TABLE ml_models (
    model_id INT PRIMARY KEY AUTO_INCREMENT,
    version VARCHAR(50),
    accuracy DECIMAL(5,2),
    trained_on DATE,
    notes TEXT
);

-- Forecasts
CREATE TABLE forecasts (
    forecast_id INT PRIMARY KEY AUTO_INCREMENT,
    restaurant_id INT,
    forecast_date DATE,
    predicted_orders INT,
    predicted_revenue DECIMAL(10,2),
    model_id INT,
    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id),
    FOREIGN KEY (model_id) REFERENCES ml_models(model_id)
);
