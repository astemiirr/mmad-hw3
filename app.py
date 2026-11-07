from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model(path):
    # Один объект модели на сервер: его не нужно копировать при каждом действии.
    return joblib.load(path)


@st.cache_data
def load_test(path):
    # Таблица — данные: cache_data возвращает отдельную копию результата.
    return pd.read_csv(path)


@st.cache_data
def predict_proba_cached(_model, X):
    # Модель исключена из хеширования; вероятности зависят от входной таблицы.
    # При изменении только порога этот результат берётся из кэша.
    return _model.predict_proba(X)[:, 1]


def prepare_batch(frame, features):
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError('В CSV не хватает столбцов: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('В CSV нет строк с данными.')
    values = frame.loc[:, features].apply(pd.to_numeric, errors='coerce')
    bad_columns = values.columns[~np.isfinite(values.to_numpy()).all(axis=0)].tolist()
    if bad_columns:
        raise ValueError('Проверьте числа и пропуски в столбцах: ' + ', '.join(bad_columns))
    if not values['is_red'].isin([0, 1]).all():
        raise ValueError('Столбец is_red должен содержать только 0 (белое) или 1 (красное).')
    return values


st.set_page_config(page_title='Подборка вин', page_icon='🍷', layout='wide')
st.title('🍷 Подборка хороших вин')
st.write('Оценка вина по результатам лабораторного анализа. Хорошее вино — оценка экспертов не ниже 7 из 10.')

try:
    model = load_model(str(APP_DIR / 'model.joblib'))
    test = load_test(str(APP_DIR / 'test.csv'))
    config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
except (OSError, ValueError) as error:
    st.error(f'Не удалось загрузить файлы приложения: {error}')
    st.stop()

features = list(model.feature_names_in_)
X_test = test.loc[:, features]
y_test = test['target'].to_numpy()
test_proba = predict_proba_cached(model, X_test)

st.sidebar.header('Настройки подборки')
threshold = st.sidebar.slider('Порог хорошего вина', min_value=0.0, max_value=1.0,
                              value=float(config['threshold']), step=0.001,
                              format='%.3f', key='threshold')
st.sidebar.caption(f"Порог по умолчанию: {config['threshold']:.4f}. Он выбран на валидации по F1.")
st.sidebar.write('Выше порог — строже отбор и меньше подборка. Ниже порог — больше хороших вин попадёт в подборку, но возрастает риск посредственных рекомендаций.')

single_tab, metrics_tab, batch_tab, weights_tab = st.tabs(
    ['Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

labels = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь, %',
    'is_red': 'Тип вина',
}

with single_tab:
    st.subheader('Оценка одного вина')
    st.caption('Диапазоны и начальные значения взяты из тестовой таблицы. Порог меняется слева.')
    values = {}
    columns = st.columns(3)
    for i, feature in enumerate(features):
        with columns[i % 3]:
            if feature == 'is_red':
                values[feature] = int(st.toggle('Красное вино', value=bool(X_test[feature].median())))
            else:
                values[feature] = st.number_input(
                    labels.get(feature, feature),
                    min_value=float(X_test[feature].min()),
                    max_value=float(X_test[feature].max()),
                    value=float(X_test[feature].median()),
                    format='%.5f' if feature == 'density' else '%.3f', key=f'input_{feature}')
    single_X = pd.DataFrame([values], columns=features)
    probability = float(predict_proba_cached(model, single_X)[0])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    if probability >= threshold:
        st.success('Решение: включить в подборку.')
    else:
        st.info('Решение: не включать в подборку при текущем пороге.')
    st.caption('Это прогноз модели, а не гарантия вкусовой оценки конкретного человека.')

with metrics_tab:
    st.subheader('Качество на отложенной тестовой выборке')
    pred = (test_proba >= threshold).astype(int)
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)
    fpr, tpr, _ = metrics.roc_curve(y_test, test_proba)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, test_proba)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    a, b, c, d = st.columns(4)
    a.metric('Precision', f'{precision:.3f}')
    b.metric('Recall', f'{recall:.3f}')
    c.metric('F1', f'{f1:.3f}')
    d.metric('Вин в подборке', f'{int(pred.sum())} / {len(pred)}')
    st.caption('Precision — доля хороших вин среди рекомендаций; recall — доля найденных хороших вин среди всех хороших в тесте.')
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Остальные', 'Хорошие']).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Матрица ошибок, порог {threshold:.3f}')
    st.pyplot(fig)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, test_proba):.3f}')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].scatter(fp / (fp + tn), recall, color='red', zorder=5, label='Текущий порог')
    axes[0].set(xlabel='False positive rate', ylabel='Recall', title='ROC-кривая', xlim=(0, 1), ylim=(0, 1.02))
    axes[1].plot(pr_recall, pr_precision, label=f'PR-AUC (AP) = {metrics.average_precision_score(y_test, test_proba):.3f}')
    axes[1].axhline(y_test.mean(), ls='--', color='gray', label='Случайное ранжирование')
    # При пустой подборке precision условно показывается как 0, как и в таблице метрик.
    axes[1].scatter(recall, precision, color='red', zorder=5, label='Текущий порог')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая', xlim=(0, 1), ylim=(0, 1.02))
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Если повысить порог, рекомендаций станет меньше: отбор строже, но часть хороших вин будет пропущена. При снижении порога подборка расширится, вместе с этим может вырасти число ошибочных рекомендаций. Precision на конкретной выборке не обязан меняться строго монотонно.')

with batch_tab:
    st.subheader('Предсказания для CSV')
    st.write('Загрузите CSV с 12 признаками, как в примере. is_red: 1 — красное, 0 — белое. Используется общий порог из боковой панели.')
    st.download_button('Скачать пример CSV', X_test.head(10).to_csv(index=False).encode('utf-8'),
                       file_name='wine_example.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            batch_X = prepare_batch(frame, features)
            batch_proba = predict_proba_cached(model, batch_X)
            result = frame.copy()
            result['probability_good'] = batch_proba
            result['prediction'] = (batch_proba >= threshold).astype(int)
            st.dataframe(result, hide_index=True)
            st.download_button('Скачать предсказания', result.to_csv(index=False).encode('utf-8'),
                               file_name='wine_predictions.csv', mime='text/csv')
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f'Не удалось обработать CSV. {error}')

with weights_tab:
    st.subheader('Какие признаки учитывает модель')
    estimator = model.steps[-1][1]
    coefficients = pd.Series(estimator.coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = np.where(coefficients >= 0, '#238b45', '#cb3c33')
    ax.barh([labels.get(f, f) for f in coefficients.index], coefficients, color=colors)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Коэффициент после стандартизации')
    ax.set_title('Вклад признаков в логарифм шансов хорошего вина')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = coefficients[coefficients > 0].nlargest(3)
    negative = coefficients[coefficients < 0].nsmallest(3)
    st.write('Сильнее всего «за»: ' + ', '.join(labels.get(f, f) for f in positive.index) + '.')
    st.write('Сильнее всего «против»: ' + ', '.join(labels.get(f, f) for f in negative.index) + '.')
    st.caption('Сравниваются изменения на одно стандартное отклонение признака при прочих равных. Связь в модели не доказывает причинного влияния.')
