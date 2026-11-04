from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


message = [
    # DELIVERY (34)
    "Delivery guy is late",
    "My delivery is taking too long",
    "Package delivery is delayed",
    "My order has not arrived yet",
    "The rider has not picked up my order",
    "The delivery partner is not moving",
    "The delivery person cannot find my address",
    "My order is stuck in transit",
    "My food arrived 45 minutes late",
    "The delivery guy took the wrong route",
    "My package was delivered to the wrong address",
    "The delivery person did not call before arriving",
    "The rider marked my order as delivered but I did not receive it",
    "The delivery person is not responding",
    "My delivery is delayed again",
    "The package arrived torn",
    "The delivery package is damaged",
    "The parcel seal was broken during delivery",
    "The food container was crushed in transit",
    "The delivery bag was leaking",
    "The soup spilled inside the delivery bag",
    "The outer package is ripped",
    "My order arrived in a broken package",
    "Some items were missing from the delivered package",
    "The package was wet when it arrived",
    "The delivery person handled my package roughly",
    "My order was left outside without any notice",
    "I received someone else's package",
    "The package was opened before delivery",
    "The delivery status has not been updated",
    "The estimated delivery time keeps changing",
    "The rider is asking for an extra delivery charge",
    "The delivery person cancelled without informing me",
    "My order was not delivered to my door",

    # PAYMENT (33)
    "My payment failed",
    "Money was deducted but the order was not confirmed",
    "I was charged twice for one order",
    "My refund has not arrived",
    "My UPI payment was unsuccessful",
    "My debit card was declined",
    "Cash on delivery is not available",
    "My wallet balance was not updated",
    "The payment page is not loading",
    "My transaction is still pending",
    "My bank account was charged but the app shows payment failed",
    "My refund is stuck",
    "My promo code is not applying at checkout",
    "The app charged me the wrong amount",
    "An extra amount was deducted from my account",
    "The invoice shows an incorrect payment amount",
    "I cannot add my credit card",
    "The payment gateway is showing an error",
    "My net banking payment failed",
    "My UPI payment is stuck on processing",
    "My cashback has not been credited",
    "Wallet payment is not working",
    "The payment timed out during checkout",
    "My preferred payment method is not accepted",
    "I was charged for a cancelled order",
    "The refund was credited to the wrong account",
    "I can see a duplicate transaction",
    "I did not receive a payment confirmation",
    "My order is placed but the payment status is pending",
    "I cannot pay using my debit card",
    "The transaction ID was not generated",
    "EMI payment is not available",
    "The payment app crashed during checkout",

    # RESTAURANT (33)
    "The food is spoiled",
    "The food is bland",
    "The food arrived cold",
    "The food tastes stale",
    "The food was undercooked",
    "The food is overcooked",
    "The restaurant sent the wrong dish",
    "Some ingredients are missing from my meal",
    "The food has too much salt",
    "The food is too spicy",
    "The portion size is too small",
    "The food smells bad",
    "My pizza is burnt",
    "My burger is soggy",
    "The fries are stale",
    "The rice is uncooked",
    "The paneer tastes sour",
    "The dal is too watery",
    "The roti is hard",
    "The salad is not fresh",
    "The food quality is poor",
    "The dish tastes different from its description",
    "The restaurant sent an incorrect order",
    "The restaurant ignored my special instructions",
    "The food is too oily",
    "I found a hair in my food",
    "The restaurant forgot the side dish",
    "The meal is completely tasteless",
    "The dessert was not prepared properly",
    "The drink tastes flat",
    "The restaurant did not make the changes I requested",
    "The food has no flavour",
    "The meal was prepared poorly"
]

labels = (
    ["Delivery"] * 34 +
    ["Payment"] * 33 +
    ["Restaurant"] * 33
)

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(message)
print(vectorizer.get_feature_names_out())


model = MultinomialNB()

model.fit(x,labels)

new_messages = ["The rider marked my order as delivered but I did not receive it"]

new_x = vectorizer.transform(new_messages)
# print(vectorizer.get_feature_names_out())
prediction = model.predict(new_x)
print(f"prediction message = {new_messages}")
print("Prediction = ",prediction)





