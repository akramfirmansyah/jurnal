from utils.adaptiveControll import AdaptiveControll

adap = AdaptiveControll()


def delaySprayController(airTemperature=None, humidity=None):
    delay = adap.FuzzyLogicNutrientPump(
        airTemperature=airTemperature, humidity=humidity
    )

    if delay is None:
        return None

    return delay
