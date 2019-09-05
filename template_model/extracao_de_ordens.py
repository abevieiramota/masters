# -*- coding: utf-8 -*-
from template_based2 import Triple
from util import load_train_dev
from gerar_base_discourse_planning import extract_orders


def test(ok, extracted):
    for i, (o, e) in enumerate(zip(ok, extracted)):
        print(i)
        if o != e:
            for ot, et in zip(o, e):
                print('ok: ', ot)
                print('ex: ', et)
                print()
        else:
            print('OK\n')


exemplos = {
        4444: [
                 (
                         Triple(subject='Wilson_Township,_Alpena_County,_Michigan', predicate='country', object='United_States'),
                         Triple(subject='Alpena_County_Regional_Airport', predicate='location', object='Wilson_Township,_Alpena_County,_Michigan'),
                         Triple(subject='Alpena_County_Regional_Airport', predicate='runwayLength', object='1533.0'),
                         Triple(subject='Alpena_County_Regional_Airport', predicate='elevationAboveTheSeaLevel_(in_metres)', object='210')
                         ),
                 (
                         Triple(subject='Alpena_County_Regional_Airport', predicate='location', object='Wilson_Township,_Alpena_County,_Michigan'),
                         Triple(subject='Wilson_Township,_Alpena_County,_Michigan', predicate='country', object='United_States'),
                         Triple(subject='Alpena_County_Regional_Airport', predicate='runwayLength', object='1533.0'),
                         Triple(subject='Alpena_County_Regional_Airport', predicate='elevationAboveTheSeaLevel_(in_metres)', object='210')
                         ),
                 (
                         Triple(subject='Alpena_County_Regional_Airport', predicate='location', object='Wilson_Township,_Alpena_County,_Michigan'),
                         Triple(subject='Wilson_Township,_Alpena_County,_Michigan', predicate='country', object='United_States'),
                         Triple(subject='Alpena_County_Regional_Airport', predicate='elevationAboveTheSeaLevel_(in_metres)', object='210'),
                         Triple(subject='Alpena_County_Regional_Airport', predicate='runwayLength', object='1533.0')
                         )
                 ],
         3333: [
                 (
                         Triple(subject='Adare_Manor', predicate='architect', object='Lewis_Nockalls_Cottingham'),
                         Triple(subject='Adare_Manor', predicate='completionDate', object='1862'),
                         Triple(subject='Adare_Manor', predicate='owner', object='J._P._McManus')
                         ),
                 (
                         Triple(subject='Adare_Manor', predicate='owner', object='J._P._McManus'),
                         Triple(subject='Adare_Manor', predicate='architect', object='Lewis_Nockalls_Cottingham'),
                         Triple(subject='Adare_Manor', predicate='completionDate', object='1862')
                         )
                 ],
        2222: [
                (
                        Triple(subject='Ballistic_(comicsCharacter)', predicate='creator', object='Michael Manley'),
                        Triple(subject='Ballistic_(comicsCharacter)', predicate='creator', object='Doug_Moench')
                        ),
                (
                        Triple(subject='Ballistic_(comicsCharacter)', predicate='creator', object='Doug_Moench'),
                        Triple(subject='Ballistic_(comicsCharacter)', predicate='creator', object='Michael Manley')
                        ),
                (
                        Triple(subject='Ballistic_(comicsCharacter)', predicate='creator', object='Michael Manley'),
                        Triple(subject='Ballistic_(comicsCharacter)', predicate='creator', object='Doug_Moench')
                        )
                ]

         }

td = load_train_dev()

for i in exemplos:

    e = td[i]
    ok = exemplos[i]
    extracted = extract_orders(e)

    print(f'{i}\n')
    test(ok, extracted)