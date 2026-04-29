from ib_insync import *

def conectar_broker():
    print("Iniciando protocolo de conexión con Interactive Brokers...")
    ib = IB()
    
    try:
        # Conectamos a tu ordenador (127.0.0.1) en el puerto de Paper Trading (7497)
        # El clientId=1 es simplemente un identificador de tu robot.
        ib.connect('127.0.0.1', 7497, clientId=1)
        print("✅ ¡Conexión exitosa a Wall Street (Simulador)!")
        
        # Pedimos el saldo de la cuenta para confirmar que estamos leyendo datos reales
        account_values = ib.accountValues()
        for val in account_values:
            # Quitamos la restricción estricta de USD. 
            # Ahora buscará 'BASE', 'EUR' o 'USD'.
            if val.tag == 'NetLiquidation' and val.currency in ['EUR', 'USD', 'BASE']:
                print(f"💰 Saldo Virtual Disponible: {float(val.value):,.2f} {val.currency}")
                
    except Exception as e:
        print(f"❌ Error crítico de conexión: {e}")
        print("-> ¿Tienes el TWS abierto y logueado en Paper Trading?")
        print("-> ¿Activaste 'Habilitar clientes Socket' en la configuración API de TWS?")
    finally:
        # Siempre cerramos la puerta al salir
        ib.disconnect()

if __name__ == "__main__":
    conectar_broker()