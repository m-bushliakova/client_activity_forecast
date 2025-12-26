# client_activity_forecast
Прогнозирование общей суммы транзакций клиента в следующем месяце для планирования маркетинговых активностей


# Цель проекта:
Прогнозировать общую сумму транзакций клиента банка в следующем месяце. Использовать эту информацию для принятия решений о маркетинговой активности.

# Задача:


# Данные:
https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation
Данные о демографии и транзакциях клиентов индийского банка.
Содержит 1 миллион транзакций 800 тыс. клиентов.
Содержит следующую информацию:
- customer age (DOB)
- location
- gender
- account balance at the time of the transaction
- transaction details
- transaction amount
- etc.

## Ключевые признаки:
CustomerID
TransactionDate
TransactionAmount (INR)

## Дополнительные признаки:
CustGender
CustLocation
CustAccountBalance

## Целевая переменная:
Целевая переменная "активность в следующем месяце" неявная, необходимо предварительно её сформулировать

## Необходимая предобработка:
- группировка транзакций по месяцу и CustomerID
- сдвиг во времени
- очистка и проверка на выбросы, пропуски

## Создание признаков:



# Стек технологий:
- python
- numpy
- matplotlib, seaborn
- scikit-learn
- 


# Как запустить

