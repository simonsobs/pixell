import solib


soda = solib.DataAccess(experiment="act",version="mr3")
imap = soda.get_map(season="s15",patch="deep56",array="pa3_150")

