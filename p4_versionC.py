import numpy as np;
from copy import deepcopy;
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc as sm


class Position:
    '''
    Une classe qui stocke une position de la puissance 4.
    Remarque: toutes les fonctions dependent du joueur actuel.
    '''

    def __init__(self, lgns = 6, cols = 7):

        # 1. Initialisation de la grille
        self.lgns = lgns;
        self.cols = cols;
        self.grille = np.zeros((lgns, cols), dtype=np.int8);    # une grille de 7 lignes et 8 colonnes.
        self.pionsParCol = [0 for i in range(cols)];    # nombre de pions par colonne.
        self.coups = 0; # nombre de coups effectues depuis le debut de la partie.

        # 2. Etat de la position
        self.gagnant = None; # None si la partie est en cours, 0 pour une partie nulle, 1 pour une partie gagnee...
        self.valeurMax = 10000; # Valeur maximale d'une position.

        # 3. Combinaisons de la position
        self.combGagnante = None;
        self.combinaisons = self.formerCombinaisons();
        self.nbrCombinaisons = len(self.combinaisons);
        self.detenteurCombinaison = [0 for i in range(self.nbrCombinaisons)];
        self.longueurCombinaison = [0 for i in range(self.nbrCombinaisons)];
        self.combinaisonsParCase = self.determinerCombinaisonsParCase();

    def afficherGrille(self):
        return None;

    def afficherPosition(self):
        '''
        Afficher la position actuelle.
        :return: None.
        '''

        print(self.grille);  # afficher la grille.
        self.afficherGrille();  # afficher la grille en images.

        if (self.gagnant == None):
            print("Joueur actuel: {}".format(self.determinerJoueurActuel()));   # afficher le joueur actuel.
        else:
            print("==================================================================================")
            print("Combinaison gagnante: {}".format(self.combGagnante));
            print("Le gagnant de cette partie est {}".format(self.gagnant));

        if(self.determinerJoueurActuel() == 2):
            print("Recherche du meilleur coup par l'IA est en cours...");

        return None;

    def determinerJoueurActuel(self):
        '''
        Determiner le joueur qui est en train de jouer.
        :return: le numero du joueur (1 pour le premier joueur et 2 pour le deuxieme).
        '''

        return (self.coups % 2) + 1;

    def coupPossible(self, col):
        '''
        Indiquer si une colonne est jouable.
        :param col: colonne a jouer (commence a 0).
        :return: true si la colonne est jouable, false si la colonne est deja pleine.
        '''

        return self.pionsParCol[col] < self.lgns;

    def ajouterPion(self, col):
        '''
        Ajouter un pion dans une colonne jouable (Attention ! : obligatoire d'utiliser coupPossible() avant).
        :param col: colonne a jouer (commence a 0)
        :return: ne retourne rien.
        '''

        # 1. Determination du joueur actuel.
        joueurActuel = self.determinerJoueurActuel();    # Determiner le joueur qui va placer son pion (1 ou 2).

        # 2. Ajout du pion dans la grille.
        self.grille[5-self.pionsParCol[col]][col] = joueurActuel;  # La grille dans le bon sens!
        self.coups += 1;

        # 3. Consequences de l'ajout du pion dans la grille.
        dernierJoueur = joueurActuel;
        self.mettreAJourDetenteurs(5-self.pionsParCol[col], col, dernierJoueur);    # mettre a jour les detenteurs des combinaisons.
        # si un alignement est fait dans une combinaison, self.gagnant vaut 1 ou 2.
        self.detecterGrillePleine();    # si la grille est pleine, self.gagnant vaut 0 !

        # 4. Modifier pions par colonne (Important de la placer apres mettreAJourDetenteurs).
        self.pionsParCol[col] += 1;

        return None;

    def formerCombinaisons(self):
        '''
        Determiner l'ensemble des combinaisons possibles pour former un alignement.
        :return: une liste de combinaisons.
        '''

        configCombinaisons = [];

        for lgn in range(self.lgns):
            for col in range(self.cols):
                if (col <= self.cols - 4):
                    configCombinaisons.append(((lgn, col), (0, 1)));
                if (lgn <= self.lgns - 4):
                    configCombinaisons.append(((lgn, col), (1, 0)));
                if (lgn <= self.lgns - 4) and (col <= self.cols - 4):
                    configCombinaisons.append(((lgn, col), (1, 1)));
                if (lgn <= self.lgns - 4) and (col >= 4 - 1):
                    configCombinaisons.append(((lgn, col), (1, -1)));

        combinaisons = [[] for i in range(len(configCombinaisons))];

        compteurCombinaison = 0;

        for ((lgnI, colI), (dy, dx)) in configCombinaisons:
            for i in range(4):
                lgn = lgnI + (i * dy);
                col = colI + (i * dx);
                combinaisons[compteurCombinaison].append((lgn, col));
            compteurCombinaison += 1;

        return combinaisons;

    def determinerCombinaisonsParCase(self):
        '''
        Creer un dictionnaire ou les cles sont les coordonnees des cases de la grille (lgn, col) et
        les entrees sont des listes de combinaisons dans les quelles intervient la case en question.
        :return: un tableau (dictionnaire)
        '''

        combinaisonsParCase = [[[] for i in range(self.cols)] for i in range(self.lgns)];   # [lgn][col]

        compteurCombinaison = 0;

        for combinaison in self.combinaisons:
            for lgn, col in combinaison:
                combinaisonsParCase[lgn][col].append(compteurCombinaison);
            compteurCombinaison += 1;

        return combinaisonsParCase;

    def mettreAJourDetenteurs(self, lgnDC, colDC, joueurDC):
        '''
        Mettre a jour les detenteurs apres chaque nouveau coup realise pour pouvoir calculer par la suite la valeur d'une
        position. (lgn et col: dernier coup joue).
        Des qu'une combinaison comporte un pion de l'adversaire elle n'est plus gagnante.
        :return: la fonction ne retourne rien.
        '''

        for combinaison in self.combinaisonsParCase[lgnDC][colDC]:
            if (self.detenteurCombinaison[combinaison] == None):
                continue;
            if (self.detenteurCombinaison[combinaison] == 0) or (self.detenteurCombinaison[combinaison] == joueurDC):
                self.detenteurCombinaison[combinaison] = joueurDC;
                self.longueurCombinaison[combinaison] += 1;

                if (self.longueurCombinaison[combinaison] >= 4):  # Detecter s'il y a une victoire pour le joueur ayant joue.
                    self.gagnant = joueurDC;
                    self.combGagnante = self.combinaisons[combinaison];

            elif (self.detenteurCombinaison[combinaison] != joueurDC):  # l'adversaire a bloque un alignement (cette combinaison
                # ne peut plus etre gagnante.
                self.detenteurCombinaison[combinaison] = None;
                self.longueurCombinaison[combinaison] = 0;

        return None;

    def detecterGrillePleine(self):
        '''
        Detecter si la grille est deja pleine (a savoir une partie nulle).
        :return: cette fonction ne retourne rien.
        '''

        if (self.coups == (self.lgns * self.cols)):
            self.gagnant = 0;  # Une partie nulle.

        return None;

    def determinerNbSeqs(self, joueur):
        '''
        Determiner a3 le nombre de combinaisons dans les quelles 3 pions sont alignes, a2...
        :param joueur: 1 pour l'humain et 2 pour l'ordinateur.
        :return: [a0, a1, a2, a3]
        '''
        seqs = [0 for k in range(4)];

        for combinaison in range(len(self.combinaisons)):
            if (self.detenteurCombinaison[combinaison] == joueur):
                seqs[self.longueurCombinaison[combinaison]] += 1;

        return seqs;

    def evaluerApproxPosition(self):
        '''
        Evaluer une position dans l'arbre de jeu revient a lui donner une valeur. (fonction d'evaluation)
        :return: valeur heristique de la position.
        '''

        joueurActuel = self.determinerJoueurActuel()   # determiner le joueur en train de jouer.

        if (self.gagnant == None):

            if joueurActuel == 1:
                autreJoueur = 2;
            else:
                autreJoueur = 1;

            poids = [0, 1, 2, 3];

            mesSeqs = self.determinerNbSeqs(joueurActuel);
            #print("Nombre de sequence de mes pions: {}".format(mesSeqs));
            maValeur = sum([s*p for s,p in zip(mesSeqs, poids)]);

            tesSeqs = self.determinerNbSeqs(autreJoueur);
            #print("Nombre de sequence des pions de l'adversaire: {}".format(tesSeqs));
            taValeur = sum([s*p for s,p in zip(tesSeqs, poids)]);

            return maValeur - taValeur;

        if (self.gagnant == 0) :
            return 0;
        elif (self.gagnant == joueurActuel):
            return self.valeurMax - self.coups;
        elif (self.gagnant != joueurActuel):
            return -self.valeurMax + self.coups;
    
    
    def afficheGrille(self, grille):
        
        """
        La fonction afficheGrille:
        prend en entree le tableau variable grille renvoie une image sous forme de grille de puissance 4 qui prend en compte les couleurs des pions joues par cahque joueur
        """
        
        imgrid=plt.imread("p4.png")#lecture de l'image
        ligne=25
        colonne=25 #ligne et colonne designent les coordonnees d'un pixel de l'image
        
        for caseC in range(7):#caseC est la coordonnee colonne de la case numerote de 0 a 6
            for caseL in range(6):
                for l in range(ligne,ligne+51,1):#caseL est la coordonnee ligne de la case  numerote de 0 a 5
                    for c in range(colonne,colonne+51,1):#ces boucles servent a colorier toutes les cases pixel par pixel sur l'image
                        if self.grille[caseL][caseC]==0:#en blanc
                            imgrid[l,c,0]=1
                            imgrid[l,c,1]=1
                            imgrid[l,c,2]=1
                        elif self.grille[caseL][caseC]==1:#en rouge
                            imgrid[l,c,0]=1
                            imgrid[l,c,1]=0
                            imgrid[l,c,2]=0
                        elif self.grille[caseL][caseC]==2:#en jaune
                            imgrid[l,c,0]=1
                            imgrid[l,c,1]=1
                            imgrid[l,c,2]=0
                
                ligne+=100 #on cree un espace entre les pixels pour la prochaine case
            ligne=25 #on reinitialise pour balayer la ligne d'en dessous
            colonne+=100
        
        sm.imsave("p4joue.png",imgrid);#on sauvegarde la nouvelle image
        grid=Image.open("p4joue.png")
        grid.show()#il s'agit simplement de lirel'image puis de la montrer



# Jeu de puissance 4

nbrParties = 0; # nombre de parties jouees.
nbrVictoires = 0;   # nombre de victoires pour le premier joueur.

compteurNoeuds = 0; # compteur des noeuds visites par IA pour evaluer une position.

def evaluerPosition(positionParent = Position(6, 7), profondeur = 5, alpha = -9999, beta = 9999):
    '''
    Resoudre recursivement une position de la puissance 4 en utilisant Negamax avec un elegage alpha-beta.
    :param positionParent: objet de la position.
    :param profondeur: profondeur de recherche.
    :param alpha: borne inferieure de la valeur.
    :param beta: borne superieure de la valeur.
    :return: valeur d'une position (presque) exacte a partir d'une valeur approximative.
    '''

    global compteurNoeuds;
    compteurNoeuds += 1;

    # 1. S'agit-il d'une position finale ? (plus aucun coup n'est possible).

    if ((positionParent.gagnant != None) or (profondeur == 0)):
        #print(positionParent.grille);
        #print(positionParent.evaluerApproxPosition());
        return positionParent.evaluerApproxPosition();  # valeur approximative (ou exacte) de la position.

    # 2. Borne superieure de la valeur de la position.

    max = positionParent.valeurMax - positionParent.coups; # Victoire immediate impossible !

    if (beta > max):
        beta = max;
        if (alpha >= beta):
            return beta;

    # 3. Position enfant de la position parent.

    for i in [3,2,4,1,5,0,6]:   # ordre d'exploration des colonnes. (ameliore alpha-beta...)
        if (positionParent.coupPossible(i)):

            positionEnfant = deepcopy(positionParent);  # creer une position enfant.

            positionEnfant.ajouterPion(i);  # ajouter un nouvel pion.

            valeur = -evaluerPosition(positionEnfant, profondeur - 1, -beta, -alpha);

            if (valeur >= beta):
                return valeur;
            if (valeur >= alpha):
                alpha = valeur;

    return alpha;

def rechercherMeilleurCoup(position, profondeur):
    '''
    Trouver le meilleur coup suivant.
    :param position: un objet de la position.
    :return: colonne a jouer.
    '''

    col = 0;
    valeurMaximale = -position.valeurMax + position.coups;  # borne inferieure de la valeur de la position.

    for i in [3, 2, 4, 1, 5, 0, 6]:
        if (position.coupPossible(i)):

            positionSuivante = deepcopy(position);

            positionSuivante.ajouterPion(i);

            valeur = -evaluerPosition(positionSuivante, profondeur);

            print("Si IA joue col{}, la valeur de jeu serait: {}".format(i, valeur));

            if (valeur > valeurMaximale):
                valeurMaximale = valeur;
                col = i;

    return col;


    
def demanderHumain(position = Position()):
    '''
    Demander le coup que veut faire l'humain.
    :return: col (le numero de la colonne [0, 6]).
    '''

    L = []  # L'ensemble des colonnes jouables pour cette position.

    for i in range(position.cols):  # Determiner l'ensemble L
        if (position.coupPossible(i) == True):
            L.append(str(i));

    entree = input("Choisissez votre colonne parmi {} :".format(L));   # Demander son coup. (Un entier)

    while (entree not in L):    # Tant que le coup n'est pas possible, redemander...
        print("Erreur. Veillez choisir une colonne compatible.");
        entree = input("Choisissez votre colonne :");
        
    col=int(entree)

    return col;




def lancerNouvellePartieIA():
    '''
    Lance le jeu de puissance 4. Humain vs Ordinateur.
    :return: None.
    '''

    position = Position(6, 7);  # Initialisation d'une nouvelle position.

    L = ['3', '4', '5'];

    print("Attention: plus la difficulte est grande, plus l'IA prend du temps pour reflechir.")
    print("==================================================================================")
    profondeur = input("Bonjour. Choisir la difficulte parmi {}: ".format(L));

    if profondeur not in L:
        print("ERREUR. Veillez recommencer.");
        return None;

    # 1. Lancement de la partie.
    
    position.afficheGrille(position.grille);#afficher l'image de la grille initiale
    
    while (position.gagnant == None): # tant que la partie est en cours.


        position.afficherPosition(); # afficher la position actuelle.

        col = 0; # colonne jouee pdt le tour actuel.

        if (position.determinerJoueurActuel() == 1):    # tour du premier ou deuxieme joueur.
            col = demanderHumain(position);   # colonne choisie par l'humain.
        else:
            col = rechercherMeilleurCoup(position, int(profondeur)); # colonne choisie par l'IA.
            print("Colonne jouee par l'ordinateur: {} (pour maximiser la valeur de jeu)".format(col));

        # coupPossible dans la colonne choisie est True.
        position.ajouterPion(col);  # ajouter un pion dans la colonne choisie par l'humain ou l'IA.
        print("==================================================================================")
        
        position.afficheGrille(position.grille);#afficher l'image de la grille durant la partie
    
    position.afficherPosition();

    # 2. Fin d'une partie.

    global nbrParties;

    nbrParties += 1;    # Incrementer le nombre de parties jouees a la fin de chaque partie.

    if (position.gagnant == 0):
        print("Partie nulle!");
    elif (position.gagnant == 1):
        print("Partie gagnee!");
        global nbrVictoires;
        nbrVictoires += 1;  # Incrementer le nombre de parties gagnees.
    else:
        print("Partie perdue!");

    print("Victoires: {}/{}".format(nbrVictoires, nbrParties)); # afficher le nombre de victoires et le nombre de parties.

    recommencer = input("Voulez-vous commencer une nouvelle partie ? (non=0 ou oui=1) : ");    # Rejouer ?

    while (recommencer != '0' and recommencer != '1'):   # attendre jusqu'a une valeur compatible...
        print("Erreur. Veillez rentrer une valeur compatible (0 ou 1).");
        recommencer = input("Voulez-vous commencer une nouvelle partie ? (non=0 ou oui=1) : ");

    if (recommencer == '1'):
        lancerNouvellePartie(); # lancer une nouvelle partie du jeu (recursivement).
    else:
        print("Aurevoir, a la prochaine !");

    return None;


def lancerNouvellePartieJoueur():
    '''
    Lance le jeu de puissance 4. Humain vs Ordinateur.
    :return: None.
    '''

    position = Position(6, 7);  # Initialisation d'une nouvelle position.


    # 1. Lancement de la partie.
    
    position.afficheGrille(position.grille);#afficher l'image de la grille initiale
    
    while (position.gagnant == None): # tant que la partie est en cours.


        position.afficherPosition(); # afficher la position actuelle.

        col = 0; # colonne jouee pdt le tour actuel.

        if (position.determinerJoueurActuel() == 1):    # tour du premier ou deuxieme joueur.
            col = demanderHumain(position);   # colonne choisie par l'humain.
        else:
            col = demanderHumain(position);

        # coupPossible dans la colonne choisie est True.
        position.ajouterPion(col);  # ajouter un pion dans la colonne choisie par l'humain ou l'IA.
        print("==================================================================================")
        
        position.afficheGrille(position.grille);#afficher l'image de la grille durant la partie
    
    position.afficherPosition();
        
        
    # 2. Fin d'une partie.

    global nbrParties;

    nbrParties += 1;    # Incrementer le nombre de parties jouees a la fin de chaque partie.

    if (position.gagnant == 0):
        print("Partie nulle!");
    elif (position.gagnant == 1):
        print("Partie gagnee!");
        global nbrVictoires;
        nbrVictoires += 1;  # Incrementer le nombre de parties gagnees.
    else:
        print("Partie perdue!");

    print("Victoires: {}/{}".format(nbrVictoires, nbrParties)); # afficher le nombre de victoires et le nombre de parties.

    recommencer = input("Voulez-vous commencer une nouvelle partie ? (non=0 ou oui=1) : ");    # Rejouer ?

    while (recommencer != '0' and recommencer != '1'):   # attendre jusqu'a une valeur compatible...
        print("Erreur. Veillez rentrer une valeur compatible (0 ou 1).");
        recommencer = input("Voulez-vous commencer une nouvelle partie ? (non=0 ou oui=1) : ");

    if (recommencer == '1'):
        lancerNouvellePartie(); # lancer une nouvelle partie du jeu (recursivement).
    else:
        print("Aurevoir, a la prochaine !");

    return None;
    
    
def lancerNouvellePartie():
    """
    Permet de determiner le mode de jeu
    """
    choix=input("Jouer contre IA=0, Jouer contre joueur = 1; Jouer contre =  ")#on demande son choix au joueur
    
    while choix !='0' and choix!='1':
        print("erreur, entrez simplement 0 ou 1")
        choix=input("Jouer contre IA=0, Jouer contre joueur = 1; Jouer contre =  ")
    
    if choix=='0':
        lancerNouvellePartieIA();
    else:
        lancerNouvellePartieJoueur();

lancerNouvellePartie();
        
   
   