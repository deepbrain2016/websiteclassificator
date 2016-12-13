'''
Created on 02/nov/2016

@author: fabrizio
'''
from unittest.main import __unittest

class OntologiaEC(set):
    '''
    classdocs
    '''


    def __init__(self):

        self.add("compra")
        self.add("carrello")
        self.add("commerce")       
        self.add("bancomat")
        self.add("carte")
        self.add("online")
        self.add("spedizione")
        self.add("pagamento")
        self.add("paga")
        self.add("biglietto")
        self.add("voucher")
        self.add("postali")
        self.add("prenotazione")
        self.add("rivendita")
        self.add("ricariche")
        self.add("contanti")
        self.add("portale")
        self.add("acquistare")
        self.add("fedelta")
        self.add("cartone")
        self.add("contante")
        self.add("prepagate")
        self.add("prepagata")
        self.add("catalogo")
        self.add("biglietti")
        self.add("rimborsi")
        self.add("dispenser")
        self.add("gratta")
        self.add("pagamenti")
        self.add("regalo")
        self.add("medicinali")
        self.add("petrolifere")
        self.add("assicurativi")
        self.add("bonifico")
        self.add("vestine")
        self.add("spedizioni")
        self.add("bancaria")
        self.add("blister")
        self.add("bancario")
        self.add("segnalibri")
        self.add("fattura")
        self.add("volantini")
        self.add("cittadino")
        self.add("fidelity")
        self.add("spesa")
        self.add("bolletta")
        self.add("bancari")
        self.add("factoring")
        self.add("acquistati")
        self.add("banconote")
        self.add("fotocopie")
        self.add("ritiro")
        self.add("tovagliolo")
        self.add("fazzoletti")
        self.add("avis")
        self.add("fotocomposizione")
        self.add("debit")
        self.add("prestito")
        self.add("imbarco")
        self.add("payday")
        self.add("tessera")
        self.add("prodotti")
        self.add("vaglia")
        self.add("tuo")
        self.add("registrazione")
        self.add("bonifici")
        self.add("fisco")
        self.add("reintegro")
        self.add("banco")
        self.add("tovaglioli")
        self.add("passaporto")
        self.add("sconto")
        self.add("moneta")
        self.add("fatturazione")
        self.add("dizionario")
        self.add("wallet")
        self.add("duplicatori")
        self.add("buoni")
        self.add("acquisto")
        self.add("sollecito")
        self.add("prenotazioni")
        self.add("negozio")
        self.add("cambiali")
        self.add("tributo")
        self.add("bollo")
        self.add("refills")
        self.add("sconti")
        self.add("prenota")
        self.add("marchio")
        self.add("riuso")
        self.add("card")
        self.add("caveau")
        self.add("latticini")
        self.add("scontrini")
        self.add("macero")
        self.add("registrati")
        self.add("journalistic")
        self.add("smarrimento")
        self.add("trasporto")
        self.add("abbonamento")
        self.add("cartoncino")
        self.add("compra")
        self.add("bolli")

    
    def paroleContestoOntologia(self,string,Ncontex):

            words=string.split(" ")
            MappaParoleTrovate=[]
            for index in range(len(words)):
                    word=words[index]

                    if word in self:
                        MappaParoleTrovate.append(index)
                        words[index]=words[index] #.upper()

                
            contestiNellaRiga=set()
            
            for parolaTrovata in MappaParoleTrovate:
                i=parolaTrovata
                end=min([i+Ncontex+1,len(words)])
                start=max([i-Ncontex,0])
                contesto=" ".join(words[start:end])
                contestiNellaRiga.add(contesto)
                
        
            return contestiNellaRiga

# 
# O=OntologiaEC()
# print O 
# string="le parole nel contesto contrassegno di credito nella busta gialla"
# print O.paroleContestoOntologia(string, 2)