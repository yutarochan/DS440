
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
'''Style '''
plt.style.use('fivethirtyeight')

funny_number=0

def plot_dec_bound(clf,X,y,name):
    global funny_number
    funny_number+=1
    for val in range(len(name)-1):
        if val<len(name)-2:
            sudo=X[:,[val,val+1]]
            clf.fit(sudo,y)
            plot_decision_regions(sudo[:1000], y[:1000], clf=clf, legend=2)
            plt.xlabel(name[val])
            plt.ylabel(name[val+1])
            plt.title('Decision Boundary '+str(clf.__class__.__name__))
            plt.savefig('image_dir/Graph19_'+str(funny_number)+str(val)+'.png',bbox_inches='tight',pad_inches=1)
            plt.close()
            #plt.show()

            sudo=X[:,[val]]
            clf.fit(sudo,y)
            plot_decision_regions(sudo[:7000], y[:7000], clf=clf, legend=2)
            plt.xlabel(name[val])
            plt.title('Single Decision Boundary '+str(clf.__class__.__name__))
            plt.savefig('image_dir/Graph20_'+str(funny_number)+str(val)+'.png',bbox_inches='tight',pad_inches=1)
            plt.close()
            #plt.show()
        else:
            sudo=X[:,[val]]
            clf.fit(sudo,y)
            plot_decision_regions(sudo[:7000], y[:7000], clf=clf, legend=2)
            plt.xlabel(name[val])
            plt.title('Single Decision Boundary '+str(clf.__class__.__name__))
            plt.savefig('image_dir/Graph21_'+str(funny_number)+str(val)+'.png',bbox_inches='tight',pad_inches=1)
            plt.close()
            #plt.show()
