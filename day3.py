# file = open('sales.csv')

# lines = file.readlines()
# file.close()

# # header
# header = lines[0].strip().split(',')
# print(header)
# m_name = ''
# t_price = 0
# total_revenue = 0
# for line in lines[1:]:
#     row = line.strip().split(',')
#     name = row[0]
#     price = int(row[1])
#     quantity = int(row[2])

#     total_price = price * quantity
#     total_revenue += total_price
#     if total_price > t_price:
#         m_name = name
#         t_price = total_price

# print("max ",m_name)
# print("total revenue ",total_revenue)


# new code
import csv
sum_amount = 0
max_amount = 0
m_product = ''
with open('sales.csv') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # print(row['product'])
        t_price = int(row['price']) * int(row['quantity'])
        sum_amount += t_price
        if(max_amount < t_price):
            max_amount = t_price
            m_product = row['product']

print('max price product =', m_product)
print('sum price =', sum_amount)