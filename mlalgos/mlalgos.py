import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class OptimizationMethod:

	def GradDesc(self,cost_fct,initial_value,learning_rate,max_iter,tolerance):
		iter_count=0;
		param=initial_value;
		cost_curve=[cost_fct(param)[0]];
		suppress_cost_warning=False;
		while np.sqrt(np.tensordot(cost_fct(param)[1],cost_fct(param)[1]))>tolerance:
			param=param-learning_rate*cost_fct(param)[1];
			cost_curve.append(cost_fct(param)[0]);
			#if (cost_curve[-1]>cost_curve[-2]) and not suppress_cost_warning:
			#	if np.sqrt(np.tensordot(cost_fct(param)[1],cost_fct(param)[1]))<0.01:
			#		suppress_cost_warning=True;
			#	else:
			#		resp=input(
			#			'Warning: Cost seems to be increasing: {old_cost:.4f} to {new_cost:.4f}.'.format(old_cost=cost_curve[-2],new_cost=cost_curve[-1])
			#			+'\nCurrent iteration: {cur_iter}'.format(cur_iter=iter_count)+'\nCurrent grad: {cur_grad}'.format(cur_grad=np.sqrt(np.tensordot(cost_fct(param)[1],cost_fct(param)[1])))
			#			+'\nTo continue enter "y", to stop enter "n", or enter new learning rate (current: {lrn_rate:e}):'.format(lrn_rate=learning_rate));
			#		if resp.lower()=='y':
			#			suppress_cost_warning=True;
			#		elif resp.lower()=='n':
			#			break;
			#		else:
			#			learning_rate=float(resp);
			iter_count+=1;
			if iter_count==max_iter:
				print('Warning: max iteration limit reached.')
				break;
		self.optimal_param=param;
		self.optimal_fct=cost_curve[-1];
		self.optimal_grad=cost_fct(param)[1];
		self.iter_count=iter_count;
		self.cost_curve=cost_curve;

			#Alternative convergence method to explore later

			#cost_initial=cost_fct(theta)[0];
			#cost_current=cost_fct(theta)[0];
			#error=1.0;
			#count=0;
			#while error>tolerance:
			#	theta=theta-learning_rate*cost_fct(theta)[1];
			#	error=(cost_current-cost_fct(theta)[0])/cost_initial;
			#	if error<0:
			#		print('Possibly learning rate too large')
			#		return None;
			#	cost_current=cost_fct(theta)[0];
	def GradDescAuto(self,cost_fct,initial_value,max_iter,tolerance):
		iter_count=0;
		param=initial_value;
		cost_curve=[cost_fct(param)[0]];
		lrn_rate=OptimizationMethod()
		lrn_rate.optimal_param=np.array([[0.1]]);
		while np.sqrt(np.tensordot(cost_fct(param)[1],cost_fct(param)[1]))>tolerance:
			def costfct_lrnrate(alph):
				fct=cost_fct(param-alph*cost_fct(param)[1])[0];
				grad=np.array(-np.tensordot(cost_fct(param-alph*cost_fct(param)[1])[1],cost_fct(param)[1]));
				grad.shape=(1,1);
				return (fct,grad)
			lrn_rate.GradDesc(costfct_lrnrate,lrn_rate.optimal_param,lrn_rate.optimal_param/100,50,1e-3)
			param=param-lrn_rate.optimal_param*cost_fct(param)[1];
			cost_curve.append(cost_fct(param)[0]);
			iter_count+=lrn_rate.iter_count;
			if iter_count==max_iter:
				print('Warning: max iteration limit reached.')
				break;
		self.optimal_param=param;
		self.optimal_fct=cost_curve[-1];
		self.optimal_grad=cost_fct(param)[1];
		self.iter_count=iter_count;
		self.cost_curve=cost_curve;

	def NormEqn(self,X,y,reg_method=None,reg_param=1.0):
		if not reg_method:
			self.optimal_param=np.matmul(np.linalg.pinv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),y));
		elif reg_method.lower()=='ridge':
			self.optimal_param=np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)+np.diag(reg_param)),np.matmul(np.transpose(X),y));
		else:
			print('Normal equation only available with no regularization or ridge regularization.')




class LinearRegressor:
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt

	def __init__(self):
		pass

	@staticmethod
	def __costfct_unreg(X,y,theta):
		m=max(len(y),1);
		fct=np.tensordot(y-np.matmul(X,theta),y-np.matmul(X,theta))/(2.0*m);
		grad=-np.matmul(np.transpose(X),y-np.matmul(X,theta))/m;
		return (fct,grad);
	
	@staticmethod
	def __costfct_ridge(X,y,reg_param,theta):
		m=max(len(y),1);
		fct=np.tensordot(y-np.matmul(X,theta),y-np.matmul(X,theta))/(2.0*m)+np.sum(reg_param*theta*theta)/(2.0*m);
		grad=-np.matmul(np.transpose(X),y-np.matmul(X,theta))/m+reg_param*theta/m;
		return (fct,grad);

	@staticmethod
	def __costfct_lasso(X,y,reg_param,theta):
		m=max(len(y),1);
		fct=np.tensordot(y-np.matmul(X,theta),y-np.matmul(X,theta))/(2.0*m)+np.sum(reg_param*np.absolute(theta))/m;
		grad=-np.matmul(np.transpose(X),y-np.matmul(X,theta))/m+reg_param*np.sign(theta)/m;
		return (fct,grad);

	@staticmethod
	def __costfct_custom(X,y,reg_fct,theta):
		m=max(len(y),1);
		fct=np.tensordot(y-np.matmul(X,theta),y-np.matmul(X,theta))/(2.0*m)+reg_fct[0](theta);
		grad=-np.matmul(np.transpose(X),y-np.matmul(X,theta))/m+reg_fct[1](theta);
		return (fct,grad);

	def fit(self, X, y,fit_intercept=True,reg_method=None,reg_param=1.0,reg_fct=None,method='GradDesc',learning_rate=0.01,theta_initial=None,tolerance=1e-10,max_iter=10000):
		X=np.array(X);
		y=np.array(y);
		num_samples=max(len(y),1);
		if len(X.shape)==0:
			self.num_features=1;
			assert num_samples==1;
		elif num_samples==1:
			self.num_features=max(X.shape);
		elif len(X.shape)==1:
			self.num_features=1;
			assert num_samples==X.shape[0];
		else:
			self.num_features=X.shape[1];
			assert num_samples==X.shape[0];


		X.shape=(num_samples,self.num_features);
		y.shape=(num_samples,1);	#Fixes issue with concatenating numpy arrays with missing dimensions


		if fit_intercept:
			X_new=np.concatenate((np.ones(shape=(num_samples,1)),X),axis=1);
			try:
				reg_param_len=len(reg_param);
			except:
				reg_param_len=1;
			finally:
				if reg_param_len==1:
					reg_param=np.concatenate(([0],reg_param*np.ones(self.num_features)));
				elif len(reg_param)==self.num_features:
					reg_param=np.concatenate(([0],reg_param));
			self.intercept=True;
		else:
			X_new=X;
			self.intercept=False;

		if method.lower()=='graddesc' or method.lower()=='graddescauto':
			
			#Compute cost function and gradient, with regularization if needed

			if reg_method==None:
				def cost_fct(theta): return LinearRegressor.__costfct_unreg(X_new,y,theta);
			elif reg_fct:
				def cost_fct(theta): return LinearRegressor.__costfct_custom(X_new,y,reg_fct,theta);
			elif reg_method.lower()=='ridge':
				def cost_fct(theta): return LinearRegressor.__costfct_ridge(X_new,y,reg_param,theta);
			elif reg_method.lower()=='lasso':
				def cost_fct(theta): return LinearRegressor.__costfct_lasso(X_new,y,reg_param,theta);
			else:
				print('If using regularization method other than ridge or lasso, must have reg_fct set')
				return None;


			#Initialize theta for gradient descent
			if not theta_initial:
				theta_initial=np.zeros(shape=(X_new.shape[1],1));

			
			#Run gradient descent
			grad_desc_costfct=OptimizationMethod();
			if method.lower()=='graddesc':
				grad_desc_costfct.GradDesc(cost_fct=cost_fct,initial_value=theta_initial,learning_rate=learning_rate,max_iter=max_iter,tolerance=tolerance);
			else:
				grad_desc_costfct.GradDescAuto(cost_fct=cost_fct,initial_value=theta_initial,max_iter=max_iter,tolerance=tolerance);
			#Output results to attribute of linear regressor
			self.params_map=grad_desc_costfct.optimal_param;
			self.grad_final=grad_desc_costfct.optimal_grad;
			self.iter_count=grad_desc_costfct.iter_count;
			self.cost_curve=grad_desc_costfct.cost_curve;

		elif method.lower()=='normeq':
			if not ((reg_method==None) or (reg_method.lower()=='ridge')):
				print('Can only use normal equation with ordinary least squares or ridge regression.')
				return None;
			else:
				norm_eqn=OptimizationMethod();
				norm_eqn.NormEqn(X_new,y,reg_method,reg_param);
				self.params_map=norm_eqn.optimal_param;

	def predict(self,X):
		X=np.array(X);
		if len(X.shape)==0:
			num_samples=1;
			assert self.num_features==1;
		elif len(X.shape)==1 and X.shape[0]==self.num_features:
			num_samples=1;
		elif len(X.shape)==1:
			num_samples=X.shape[0];
			assert self.num_features==1;
		else:
			num_samples=X.shape[0];
			assert self.num_features==X.shape[1];
		X.shape=(num_samples,self.num_features);
		if not self.intercept:
			return np.matmul(X,self.params_map);
		else:
			X_new=np.concatenate((np.ones(shape=(X.shape[0],1)),X),axis=1);
			return np.matmul(X_new,self.params_map);

def sigmoid(z): return 1.0/(1.0+np.exp(-z));

class LogisticRegressor:
	def __init__(self):
		pass

	@staticmethod
	def __costfct_unreg(X,y,theta):
		m=max(y.shape[0],1);
		fct=(-np.tensordot(y,np.log(sigmoid(np.matmul(X,theta))))-np.tensordot(1.0-y,1.0-np.log(sigmoid(np.matmul(X,theta)))))/m;
		grad=-np.matmul(np.transpose(X),y-sigmoid(np.matmul(X,theta)))/m;
		return (fct,grad);
	
	@staticmethod
	def __costfct_ridge(X,y,reg_param,theta):
		m=max(y.shape[0],1);
		fct=(-np.tensordot(y,np.log(sigmoid(np.matmul(X,theta))))-np.tensordot(1.0-y,1.0-np.log(sigmoid(np.matmul(X,theta)))))/m+np.sum(reg_param*theta*theta)/(2.0*m);
		grad=-np.matmul(np.transpose(X),y-sigmoid(np.matmul(X,theta)))/m+reg_param*theta/m;
		return (fct,grad);

	@staticmethod
	def __costfct_lasso(X,y,reg_param,theta):
		m=max(y.shape[0],1);
		fct=(-np.tensordot(y,np.log(sigmoid(np.matmul(X,theta))))-np.tensordot(1.0-y,1.0-np.log(sigmoid(np.matmul(X,theta)))))/m+np.sum(reg_param*np.absolute(theta))/m;
		grad=-np.matmul(np.transpose(X),y-sigmoid(np.matmul(X,theta)))/m+reg_param*np.sign(theta)/m;
		return (fct,grad);

	@staticmethod
	def __costfct_custom(X,y,reg_fct,theta):
		m=max(y.shape[0],1);
		fct=(-np.tensordot(y,np.log(sigmoid(np.matmul(X,theta))))-np.tensordot(1.0-y,1.0-np.log(sigmoid(np.matmul(X,theta)))))/m+reg_fct[0](theta);
		grad-np.matmul(np.transpose(X),y-sigmoid(np.matmul(X,theta)))/m+reg_fct[1](theta);
		return (fct,grad);

	def fit(self, X, y,reg_method=None,reg_param=1.0,reg_fct=None,method='GradDesc',learning_rate=0.01,theta_initial=None,tolerance=1e-10,max_iter=10000):
		X=np.array(X);
		y=np.array(y);
		if len(y.shape)==1:
			num_samples=1
			num_classes=y.shape[0];
		else:
			num_samples=y.shape[0];
			num_classes=y.shape[1];

		if len(X.shape)==0:
			self.num_features=1;
			assert num_samples==1;
		elif num_samples==1:
			self.num_features=max(X.shape);
		elif len(X.shape)==1:
			self.num_features=1;
			assert num_samples==X.shape[0];
		else:
			self.num_features=X.shape[1];
			assert num_samples==X.shape[0];


		X.shape=(num_samples,self.num_features);
		y.shape=(num_samples,num_classes);	#Fixes issue with concatenating numpy arrays with missing dimensions

		X_new=np.concatenate((np.ones(shape=(num_samples,1)),X),axis=1);
		try:
			reg_param_len=len(reg_param);
		except:
			reg_param_len=1;
		finally:
			if reg_param_len==1:
				reg_param=np.concatenate(([0],reg_param*np.ones(self.num_features)));
			elif reg_param_len==self.num_features:
				reg_param=np.concatenate(([0],reg_param));

		if method.lower()=='graddesc' or method.lower()=='graddescauto':

			#Compute cost function and gradient, with regularization if needed
			if reg_method==None:
				def cost_fct(theta): return LogisticRegressor.__costfct_unreg(X_new,y,theta);
			elif reg_fct:
				def cost_fct(theta): return LogisticRegressor.__costfct_custom(X_new,y,reg_fct,theta);
			elif reg_method.lower()=='ridge':
				def cost_fct(theta): return LogisticRegressor.__costfct_ridge(X_new,y,reg_param,theta);
			elif reg_method.lower()=='lasso':
				def cost_fct(theta): return LogisticRegressor.__costfct_lasso(X_new,y,reg_param,theta);
			else:
				print('If using regularization method other than ridge or lasso, must have reg_fct set')
				return None;


			#Initialize theta for gradient descent
			if not theta_initial:
				theta_initial=np.zeros(shape=(self.num_features+1,num_classes));
				
			#Run gradient descent
			grad_desc_costfct=OptimizationMethod();
			if method.lower()=='graddesc':
				grad_desc_costfct.GradDesc(cost_fct=cost_fct,initial_value=theta_initial,learning_rate=learning_rate,max_iter=max_iter,tolerance=tolerance);
			else:
				grad_desc_costfct.GradDescAuto(cost_fct=cost_fct,initial_value=theta_initial,max_iter=max_iter,tolerance=tolerance);
			
			#Output results to attribute of linear regressor
			self.params_map=grad_desc_costfct.optimal_param;
			self.grad_final=grad_desc_costfct.optimal_grad;
			self.iter_count=grad_desc_costfct.iter_count;
			self.cost_curve=grad_desc_costfct.cost_curve;


	def predict(self,X):
		X=np.array(X);
		if len(X.shape)==0:
			num_samples=1;
			assert self.num_features==1;
		elif len(X.shape)==1 and X.shape[0]==self.num_features:
			num_samples=1;
		elif len(X.shape)==1:
			num_samples=X.shape[0];
			assert self.num_features==1;
		else:
			num_samples=X.shape[0];
			assert self.num_features==X.shape[1];
		X.shape=(num_samples,self.num_features);
			
		X_new=np.concatenate((np.ones(shape=(num_samples,1)),X),axis=1);
		weights=sigmoid(np.matmul(X_new,self.params_map));
		weight=np.max(weights,axis=1);
		level=np.argmax(weights,axis=1);
		return (level,weight,weights)
