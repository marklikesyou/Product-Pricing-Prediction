import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_price_tiers(x):
    if len(x) < 5:
        return pd.Series(['Medium'] * len(x))
    
    ranks = x.rank(pct=True)
    tiers = pd.Series(index=ranks.index, data='Medium')
    tiers[ranks <= 0.2] = 'Low'
    tiers[(ranks > 0.2) & (ranks <= 0.4)] = 'Medium-Low'
    tiers[(ranks > 0.6) & (ranks <= 0.8)] = 'Medium-High'
    tiers[ranks > 0.8] = 'High'
    return tiers

def load_and_preprocess_data(sample_size=10000):
    df = pd.read_csv('dataset.csv', 
                    usecols=['prices.amountMin', 'prices.condition', 'prices.merchant', 
                            'prices.dateSeen', 'brand', 'categories', 'primaryCategories'])
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    df = df.dropna(subset=['prices.amountMin'])
    
    df['first_date_seen'] = pd.to_datetime(
        df['prices.dateSeen'].str.split(',').str[0], 
        format='%Y-%m-%dT%H:%M:%SZ'
    )
    
    df['brand'].fillna('Unknown', inplace=True)
    df['prices.merchant'].fillna('Unknown', inplace=True)
    df['prices.condition'].fillna('New', inplace=True)
    
    df['brand_tier'] = df.groupby('brand')['prices.amountMin'].transform(create_price_tiers)
    df['merchant_tier'] = df.groupby('prices.merchant')['prices.amountMin'].transform(create_price_tiers)
    
    df['primary_category'] = df['primaryCategories']
    df['secondary_category'] = df['categories'].str.split(',').str[1].fillna('Unknown')
    
    df['brand_merchant'] = df['brand'] + "_" + df['prices.merchant']
    df['category_condition'] = df['primary_category'] + "_" + df['prices.condition']
    df['brand_category'] = df['brand'] + "_" + df['primary_category']
    
    for col in ['brand', 'prices.merchant', 'primary_category', 'secondary_category']:
        value_counts = df[col].value_counts(normalize=True)
        rare_categories = value_counts[value_counts < 0.01].index
        df[col] = df[col].replace(rare_categories, 'Other')
    
    df['month'] = df['first_date_seen'].dt.month
    df['is_weekend'] = df['first_date_seen'].dt.dayofweek.isin([5, 6]).astype(int)
    
    month_to_season = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df['season'] = df['first_date_seen'].dt.month.map(month_to_season)
    
    df['price'] = np.log1p(df['prices.amountMin'])
    
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR)))]
    
    return df

def engineer_features(df):
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    df['price_category_zscore'] = df.groupby('primary_category')['price'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    df['price_category_quantile'] = df.groupby('primary_category')['price'].transform(
        lambda x: pd.qcut(x, q=5, labels=['VL', 'L', 'M', 'H', 'VH'], duplicates='drop')
    )
    df['merchant_count_per_category'] = df.groupby('primary_category')['prices.merchant'].transform('nunique')
    df['brand_count_per_category'] = df.groupby('primary_category')['brand'].transform('nunique')
    df['brand_share'] = df.groupby(['primary_category', 'brand'])['prices.merchant'].transform('count') / \
                       df.groupby('primary_category')['prices.merchant'].transform('count')
    
    df['category_condition'] = df['primary_category'] + '_' + df['prices.condition'].fillna('Unknown')
    df['merchant_season'] = df['prices.merchant'] + '_' + df['season'].fillna('Unknown')
    df['brand_condition'] = df['brand'] + '_' + df['prices.condition'].fillna('Unknown')
    df['category_median_price'] = df.groupby('primary_category')['price'].transform('median')
    df['brand_median_price'] = df.groupby('brand')['price'].transform('median')
    df['price_to_category_median'] = df['price'] / df['category_median_price']
    df['price_to_brand_median'] = df['price'] / df['brand_median_price']
    
 
    def remove_outliers(x):
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return np.clip(x, lower, upper)
    
    df['price'] = df.groupby('primary_category')['price'].transform(remove_outliers)
    
    feature_cols = [
        'month', 'is_weekend', 'month_sin', 'month_cos', 
        'price_category_zscore', 'merchant_count_per_category',
        'brand_count_per_category', 'brand_share',
        'category_median_price', 'brand_median_price',
        'price_to_category_median', 'price_to_brand_median',
        'brand', 'prices.merchant', 'primary_category', 'secondary_category',
        'brand_tier', 'merchant_tier', 'season', 'category_condition',
        'merchant_season', 'brand_condition', 'price_category_quantile'
    ]
    
    return df[feature_cols], df['price']

def create_ml_pipeline():
    numeric_features = [
        'month', 'is_weekend', 'month_sin', 'month_cos',
        'price_category_zscore', 'merchant_count_per_category',
        'brand_count_per_category', 'brand_share',
        'category_median_price', 'brand_median_price',
        'price_to_category_median', 'price_to_brand_median'
    ]
    
    categorical_features = [
        'brand', 'prices.merchant', 'primary_category', 'secondary_category',
        'brand_tier', 'merchant_tier', 'season', 'category_condition',
        'merchant_season', 'brand_condition', 'price_category_quantile'
    ]
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Other')),
        ('encoder', OneHotEncoder(sparse_output=False, min_frequency=0.01,
                                handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        max_samples=0.9,
        random_state=42
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        random_state=42
    )
    
    et = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=30,
        min_samples_split=2,
        bootstrap=True,
        random_state=42
    )
    
    estimators = [
        ('rf', rf),
        ('gb', gb),
        ('et', et)
    ]
    
    final_estimator = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42
    )
    
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectFromModel(
            RandomForestRegressor(n_estimators=200, random_state=42),
            threshold='1.25*mean'
        )),
        ('regressor', stacking)
    ])
    
    param_grid = {
        'regressor__rf__max_depth': [35, 40],
        'regressor__rf__max_samples': [0.8, 0.9],
        'regressor__gb__learning_rate': [0.03, 0.05],
        'regressor__gb__n_estimators': [250, 300]
    }
    
    scoring = {
        'r2': 'r2',
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_root_mean_squared_error': 'neg_root_mean_squared_error'
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring=scoring,
        refit='r2',
        verbose=1,
        return_train_score=True
    )
    
    return grid_search

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    correlation, p_value = stats.pearsonr(y_test, y_pred)
    
    y_pred_cv = cross_val_score(model.best_estimator_, X_test, y_test, cv=5, scoring='r2')
    mae_cv = cross_val_score(model.best_estimator_, X_test, y_test, cv=5, scoring='neg_mean_absolute_error')
    rmse_cv = cross_val_score(model.best_estimator_, X_test, y_test, cv=5, scoring='neg_root_mean_squared_error')
    
    feature_names = []
    if hasattr(model.best_estimator_.named_steps['preprocessor'], 'get_feature_names_out'):
        feature_names = model.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
    
    selected_features_mask = model.best_estimator_.named_steps['selector'].get_support()
    selected_features = feature_names[selected_features_mask]
    
    return {
        'metrics': {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation,
            'p_value': p_value
        },
        'cv_scores': {
            'r2': y_pred_cv,
            'mae': mae_cv,
            'rmse': rmse_cv
        },
        'selected_features': selected_features
    }

def analyze_feature_importance(model, feature_names):
    rf_estimator = model.best_estimator_.named_steps['regressor'].estimators_[0][1]
    importances = rf_estimator.feature_importances_
    feature_names = feature_names[:len(importances)]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features in Price Prediction', fontsize=14, pad=20)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    total = feature_importance['importance'].sum()
    for i, v in enumerate(feature_importance.head(15)['importance']):
        plt.text(v, i, f' {v/total*100:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_importance

def generate_price_insights(df, model, X_test, y_test, y_pred):
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig('residual_plot.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    insights = {
        'mean_pred': np.mean(y_pred),
        'median_pred': np.median(y_pred),
        'pred_std': np.std(y_pred),
        'mean_residual': np.mean(residuals),
        'residual_std': np.std(residuals)
    }
    
    return insights

def main():
    df = load_and_preprocess_data()
    X, y = engineer_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_ml_pipeline()
    model.fit(X_train, y_train)
    
    evaluation_results = evaluate_model(model, X_test, y_test)
    y_pred = model.predict(X_test)
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    feature_names = list(numeric_features) + list(categorical_features)
    
    feature_importance = analyze_feature_importance(model, feature_names)
    insights = generate_price_insights(df, model, X_test, y_test, y_pred)

if __name__ == "__main__":
    main()
