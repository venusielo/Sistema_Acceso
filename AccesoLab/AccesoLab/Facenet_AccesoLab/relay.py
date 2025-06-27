import pyhid_usb_relay


def off():
    relay = pyhid_usb_relay.find()
    # If relay state is on (1) then we can turn it off.
    if relay.state:
        relay.toggle_state(1)
    else:
        pass
        


def on():
    relay = pyhid_usb_relay.find()
    # If relay state is not on (1) then we can turn it on.
    if not relay.state:
        relay.toggle_state(1)
        off() #Llamamos dentro de la funcion ya que asi funciona el abrir y cerrar la puerta bien
    else:
        pass

